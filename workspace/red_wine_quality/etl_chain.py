import pandas as pd
from pydantic import BaseModel, Field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union, Iterator, Tuple
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler 
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances 
import numpy as np 
import time 

import re
from collections import defaultdict
from scipy.stats import fisher_exact 

from pathlib import Path

# --- Pydantic Input/Output Schemas ---

class BaseETLData(BaseModel):
    X: pd.DataFrame
    y: Optional[pd.Series] = None

    class Config:
        arbitrary_types_allowed = True

class TransformedData(BaseETLData):
    pass

class FittedState(BaseModel):
    state: Dict[str, Any] = Field(default_factory=dict)

class StreamMessage(BaseModel):
    step_name: str
    status: str 
    message: Optional[str] = None
    progress: Optional[float] = None 
    data_preview: Optional[Dict[str, Any]] = None

# --- Base Runnable Class ---

InputType = TypeVar('InputType', bound=BaseETLData)
OutputType = TypeVar('OutputType', bound=BaseETLData)
StateType = TypeVar('StateType', bound=FittedState)

class ETLRunnable(Generic[InputType, OutputType, StateType]):
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._fitted_state: Optional[StateType] = None

    def fit(self, data: InputType) -> 'ETLRunnable[InputType, OutputType, StateType]':
        validated_data = self._validate_input(data, self._get_input_model_fit())
        self._fitted_state = self._fit(validated_data)
        return self

    def transform(self, data: InputType, fitted_state: Optional[StateType] = None) -> OutputType:
        validated_data = self._validate_input(data, self._get_input_model_transform())
        current_state = fitted_state or self._fitted_state
        if current_state is None:
            raise RuntimeError(f"Runnable '{self.name}' has not been fitted or no state provided.")
        transformed_data = self._transform(validated_data, current_state)
        return self._validate_output(transformed_data, self._get_output_model_transform())

    def fit_transform(self, data: InputType) -> OutputType:
        self.fit(data)
        return self.transform(data)

    def transform_stream(self, data: InputType, fitted_state: Optional[StateType] = None) -> Iterator[Union[StreamMessage, OutputType]]:
        validated_data = self._validate_input(data, self._get_input_model_transform())
        current_state = fitted_state or self._fitted_state
        if current_state is None:
            yield StreamMessage(step_name=self.name, status="error", message="Runnable has not been fitted or no state provided.")
            raise RuntimeError(f"Runnable '{self.name}' has not been fitted or no state provided.")

        yield StreamMessage(step_name=self.name, status="starting", message="Transformation starting.")
        final_output = None
        try:
            for item in self._transform_stream(validated_data, current_state):
                if isinstance(item, StreamMessage):
                    yield item
                else:
                    final_output = item
        except Exception as e:
            yield StreamMessage(step_name=self.name, status="error", message=f"Error during transform_stream: {str(e)}")
            raise

        if final_output is None:
            yield StreamMessage(step_name=self.name, status="error", message="Transformation stream did not yield a final output.")
            raise ValueError(f"{self.name}._transform_stream did not yield a final OutputType result.")

        validated_final_output = self._validate_output(final_output, self._get_output_model_transform())
        yield StreamMessage(step_name=self.name, status="completed", message="Transformation complete.")
        yield validated_final_output

    def _fit(self, data: InputType) -> StateType:
        raise NotImplementedError

    def _transform(self, data: InputType, state: StateType) -> OutputType:
        raise NotImplementedError

    def _transform_stream(self, data: InputType, state: StateType) -> Iterator[Union[StreamMessage, OutputType]]:
        yield StreamMessage(step_name=self.name, status="in_progress", message="Executing non-streaming transform.")
        result = self._transform(data, state)
        yield result

    def _get_input_model_fit(self) -> type[BaseModel]:
        return BaseETLData
    def _get_input_model_transform(self) -> type[BaseModel]:
        return BaseETLData
    def _get_output_model_transform(self) -> type[BaseModel]:
        return TransformedData
    def _get_state_model(self) -> type[BaseModel]: 
        return FittedState

    def _validate_input(self, data: Any, model: type[BaseModel]) -> BaseModel:
        if isinstance(data, model): return data
        if isinstance(data, dict): return model(**data)
        raise TypeError(f"Input for '{self.name}' invalid: expected {model.__name__} or dict, got {type(data)}")

    def _validate_output(self, data: Any, model: type[BaseModel]) -> BaseModel:
        if isinstance(data, model): return data
        if isinstance(data, dict) and hasattr(model, 'X') and hasattr(model, 'y'): 
             return model(**data)
        if isinstance(data, TransformedData) and issubclass(model, TransformedData):
            return data
        raise TypeError(f"Output for '{self.name}' invalid: expected {model.__name__}, got {type(data)}. Data: {str(data)[:100]}")


    def __or__(self, other: 'ETLRunnable') -> 'ETLChain':
        if isinstance(self, ETLChain):
            return ETLChain(self.steps + [other])
        return ETLChain([self, other])

    @property
    def fitted_state(self) -> Optional[StateType]:
        return self._fitted_state

    def set_fitted_state(self, state: StateType):
        expected_state_type = self._get_state_model()
        if not isinstance(state, expected_state_type):
            try:
                if isinstance(state, dict):
                    state = expected_state_type(**state)
                elif isinstance(state, FittedState) and expected_state_type != FittedState:
                     state = expected_state_type(**state.dict())
                else:
                    raise TypeError() 
            except Exception as e:
                 raise TypeError(f"State for '{self.name}' must be {expected_state_type.__name__}, got {type(state)}. Cast error: {e}")
        self._fitted_state = state

# --- ETL Chain Class ---
class ETLChain(ETLRunnable[BaseETLData, TransformedData, FittedState]):
    def __init__(self, steps: List[ETLRunnable]):
        super().__init__(name="ETLChain")
        self.steps = steps

    def _fit(self, data: BaseETLData) -> FittedState:
        current_data_x = data.X.copy()
        current_data_y = data.y.copy() if data.y is not None else None
        all_step_states_dict = {} 
        for i, step in enumerate(self.steps):
            step_input_model = step._get_input_model_fit()
            
            # Determine the correct input data structure for the current step's fit method
            if step_input_model == NameWordFeaturesInput or isinstance(step, NameWordFeatures):
                if current_data_y is None:
                    raise ValueError(f"Input data for fitting step '{step.name}' is missing 'y' series, but it's required.")
                step_input_data = NameWordFeaturesInput(X=current_data_x, y=current_data_y)
            elif step_input_model == ServiceFeaturesComponentInput or isinstance(step, ServiceFeaturesComponent):
                 if current_data_y is None: # y is not strictly needed for fit here but passed for model consistency
                    # Create a dummy y if it's absolutely required by the model but not available
                    # For ServiceFeaturesComponent, y is optional for fit, so this might not be an issue.
                    # However, if its _get_input_model_fit strictly requires y, this needs handling.
                    # For now, assume y is available or the component handles its absence.
                     pass # y is optional for ServiceFeaturesComponent fit input model
                 step_input_data = ServiceFeaturesComponentInput(X=current_data_x, y=current_data_y)
            else: # Default to BaseETLData
                step_input_data = BaseETLData(X=current_data_x, y=current_data_y)
            
            print(f"Fitting step in chain: {step.name}")
            step.fit(step_input_data) 
            if step.fitted_state:
                all_step_states_dict[f"{step.name}_{i}_state_dict"] = step.fitted_state.dict()
            
            # Transform the data using the just-fitted step to prepare input for the *next* step's fit
            if step.fitted_state is not None:
                # Determine input for transform based on the step's transform input model
                transform_input_model = step._get_input_model_transform()
                if transform_input_model == NameWordFeaturesInput or isinstance(step, NameWordFeatures):
                    transform_input_data = NameWordFeaturesInput(X=current_data_x, y=current_data_y)
                elif transform_input_model == ServiceFeaturesComponentInput or isinstance(step, ServiceFeaturesComponent):
                     transform_input_data = ServiceFeaturesComponentInput(X=current_data_x, y=current_data_y)
                else:
                    transform_input_data = BaseETLData(X=current_data_x, y=current_data_y)

                transformed_for_next_fit = step.transform(transform_input_data)
                current_data_x = transformed_for_next_fit.X
                if transformed_for_next_fit.y is not None: 
                    current_data_y = transformed_for_next_fit.y
        return FittedState(state=all_step_states_dict)

    def _transform(self, data: BaseETLData, state: FittedState) -> TransformedData:
        current_data_x = data.X.copy()
        current_data_y = data.y.copy() if data.y is not None else None

        for i, step in enumerate(self.steps):
            if step.fitted_state is None:
                step_state_dict_key = f"{step.name}_{i}_state_dict"
                if step_state_dict_key in state.state:
                    specific_step_state_model = step._get_state_model()
                    try:
                        step.set_fitted_state(specific_step_state_model(**state.state[step_state_dict_key]))
                    except Exception as e:
                        raise RuntimeError(f"Failed to load/set state for step '{step.name}' from chain state: {e}")
            if step.fitted_state is None:
                 raise RuntimeError(f"Step '{step.name}' in chain was not properly fitted or its state is missing from chain state dict.")
            
            step_input_model = step._get_input_model_transform()
            if step_input_model == NameWordFeaturesInput or isinstance(step, NameWordFeatures):
                 step_input_data = NameWordFeaturesInput(X=current_data_x, y=current_data_y)
            elif step_input_model == ServiceFeaturesComponentInput or isinstance(step, ServiceFeaturesComponent):
                 step_input_data = ServiceFeaturesComponentInput(X=current_data_x, y=current_data_y) 
            else: 
                step_input_data = BaseETLData(X=current_data_x, y=current_data_y)
            
            transformed_output = step.transform(step_input_data)
            current_data_x = transformed_output.X
            if transformed_output.y is not None:
                current_data_y = transformed_output.y
        return TransformedData(X=current_data_x, y=current_data_y)

    def _transform_stream(self, data: BaseETLData, state: FittedState) -> Iterator[Union[StreamMessage, TransformedData]]:
        current_data_x = data.X.copy()
        current_data_y = data.y.copy() if data.y is not None else None
        final_output_data = None

        yield StreamMessage(step_name=self.name, status="starting", message="Chain transformation starting.")

        for i, step in enumerate(self.steps):
            yield StreamMessage(
                step_name=self.name,
                status="in_progress",
                message=f"Starting sub-step: {step.name} ({i+1}/{len(self.steps)})"
            )

            if step.fitted_state is None:
                step_state_dict_key = f"{step.name}_{i}_state_dict"
                if step_state_dict_key in state.state:
                    specific_step_state_model = step._get_state_model()
                    try:
                        step.set_fitted_state(specific_step_state_model(**state.state[step_state_dict_key]))
                    except Exception as e:
                        err_msg = f"Failed to load/set state for step '{step.name}' for streaming: {e}"
                        yield StreamMessage(step_name=self.name, status="error", message=err_msg)
                        raise RuntimeError(err_msg)

            if step.fitted_state is None:
                err_msg = f"Step '{step.name}' in chain was not properly fitted or its state is missing for streaming."
                yield StreamMessage(step_name=self.name, status="error", message=err_msg)
                raise RuntimeError(err_msg)
            
            step_input_model = step._get_input_model_transform()
            if step_input_model == NameWordFeaturesInput or isinstance(step, NameWordFeatures):
                 step_input_data = NameWordFeaturesInput(X=current_data_x, y=current_data_y)
            elif step_input_model == ServiceFeaturesComponentInput or isinstance(step, ServiceFeaturesComponent):
                step_input_data = ServiceFeaturesComponentInput(X=current_data_x, y=current_data_y)
            else:
                step_input_data = BaseETLData(X=current_data_x, y=current_data_y)

            step_output_data = None
            gen = step.transform_stream(step_input_data)
            for item in gen:
                if isinstance(item, StreamMessage):
                    item.message = f"[Chain -> {step.name}] {item.message}" 
                    yield item
                else: 
                    step_output_data = item
            
            if step_output_data is None:
                err_msg = f"Sub-step {step.name} in chain did not yield final data."
                yield StreamMessage(step_name=self.name, status="error", message=err_msg)
                raise ValueError(err_msg)

            current_data_x = step_output_data.X
            if step_output_data.y is not None: 
                current_data_y = step_output_data.y

            yield StreamMessage(
                step_name=self.name,
                status="in_progress",
                message=f"Completed sub-step: {step.name}. Shape after: {current_data_x.shape}"
            )

        final_output_data = TransformedData(X=current_data_x, y=current_data_y)
        yield StreamMessage(step_name=self.name, status="completed", message="Chain transformation complete.")
        yield final_output_data


    @property
    def fitted_states_from_steps(self) -> List[Optional[FittedState]]:
        return [step.fitted_state for step in self.steps]

    def set_fitted_states_on_steps(self, states: List[Optional[FittedState]]):
        if len(states) != len(self.steps):
            raise ValueError("Number of states must match number of steps.")
        for step, state_to_set in zip(self.steps, states):
            if state_to_set is not None:
                step.set_fitted_state(state_to_set)


# --- Specific ETL Components ---

# 1. RawTransformer Component (Revised categorical encoding)
class RawTransformerState(FittedState):
    service_cols: List[str]
    # Store the learned categories themselves for consistent .cat.codes application
    homeplanet_categories: Optional[List[str]] = None
    destination_categories: Optional[List[str]] = None
    deck_categories: Optional[List[str]] = None
    side_categories: Optional[List[str]] = None
    max_spend_categories: Optional[List[str]] = None # For max_spend_category
    age_median: Optional[float] = None # To store median of Age from training data

class RawTransformerInput(BaseETLData): pass
class RawTransformerOutput(TransformedData): pass

class RawTransformer(ETLRunnable[RawTransformerInput, RawTransformerOutput, RawTransformerState]):
    def __init__(self, service_cols: List[str]):
        super().__init__(name="RawTransformer")
        self.service_cols = service_cols

    def _fit(self, data: RawTransformerInput) -> RawTransformerState:
        X = data.X.copy()
        state = RawTransformerState(service_cols=self.service_cols)

        if 'HomePlanet' in X.columns:
            state.homeplanet_categories = list(X['HomePlanet'].astype('category').cat.categories)
        if 'Destination' in X.columns:
            state.destination_categories = list(X['Destination'].astype('category').cat.categories)
        
        if 'Cabin' in X.columns and X['Cabin'].notna().any():
            cabin_parts = X["Cabin"].str.split("/", expand=True)
            if cabin_parts.shape[1] == 3:
                state.deck_categories = list(cabin_parts[0].astype('category').cat.categories)
                state.side_categories = list(cabin_parts[2].astype('category').cat.categories)
        
        if 'Age' in X.columns:
            state.age_median = X['Age'].median()

        # Fit max_spend_category categories
        temp_X_fit = X.copy() 
        valid_service_cols_fit = [sc for sc in self.service_cols if sc in temp_X_fit.columns]
        
        # Ensure service columns are numeric before idxmax for fitting categories
        for col in valid_service_cols_fit:
            temp_X_fit[col] = pd.to_numeric(temp_X_fit[col], errors='coerce').fillna(0)

        if valid_service_cols_fit:
            all_numeric_for_idxmax = all(pd.api.types.is_numeric_dtype(temp_X_fit[col]) for col in valid_service_cols_fit)
            if all_numeric_for_idxmax and not temp_X_fit[valid_service_cols_fit].empty:
                # Check if all values in a row can be zero, idxmax might raise error or pick first if all same
                if not temp_X_fit[valid_service_cols_fit].apply(lambda row: (row == 0).all(), axis=1).all():
                    max_spend_series = temp_X_fit[valid_service_cols_fit].idxmax(axis=1)
                    state.max_spend_categories = list(max_spend_series.astype('category').cat.categories)
                else:
                    print(f"Warning for {self.name} fit: All service col values are zero for some rows, idxmax may be ambiguous. Max_spend_categories might be limited.")
                    # Handle case where all rows might be all zeros for service columns
                    # state.max_spend_categories might remain None or be based on column names if that's desired fallback
                    state.max_spend_categories = valid_service_cols_fit # Fallback to service col names if all idxmax are ambiguous
            else:
                 print(f"Warning for {self.name} fit: Not all service columns are numeric for idxmax or df is empty. Max_spend_categories may not be set.")


        return state

    def _transform(self, data: RawTransformerInput, state: RawTransformerState) -> RawTransformerOutput:
        X_df = data.X.copy()
        y_series = data.y.copy() if data.y is not None else None

        if "PassengerId" in X_df.columns: # Drop PassengerId if it exists
            X_df = X_df.drop(columns=["PassengerId"])

        for col in ["CryoSleep", "VIP"]:
            if col in X_df.columns:
                if X_df[col].dtype == object: 
                    X_df[col] = X_df[col].astype(str).str.strip().str.lower()
                    X_df[col] = X_df[col].map({"true": True, "false": False, '0': False, '1': True})
                # Fill NaNs with False for boolean conversion, as in etl.py's implicit behavior or explicit fill
                X_df[col] = X_df[col].fillna(False).astype(bool)

        if 'HomePlanet' in X_df.columns and state.homeplanet_categories is not None:
            X_df['HomePlanet'] = X_df['HomePlanet'].astype('category').cat.set_categories(state.homeplanet_categories).cat.codes
        elif 'HomePlanet' in X_df.columns: # Column exists but no categories learned (e.g. fit data had all NaNs)
            X_df['HomePlanet'] = -1 # Default code

        if 'Destination' in X_df.columns and state.destination_categories is not None:
            X_df['Destination'] = X_df['Destination'].astype('category').cat.set_categories(state.destination_categories).cat.codes
        elif 'Destination' in X_df.columns:
            X_df['Destination'] = -1
            
        valid_service_cols = [sc for sc in state.service_cols if sc in X_df.columns]
        for col in valid_service_cols: # Fill NaNs with 0 for service columns
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
        
        if valid_service_cols:
            X_df["total_spent"] = X_df[valid_service_cols].sum(axis=1)
            
            temp_service_df_for_idxmax = X_df[valid_service_cols].copy() # Use a copy
            # Ensure all columns for idxmax are numeric
            all_numeric_transform = True
            for col_idx in temp_service_df_for_idxmax.columns:
                if not pd.api.types.is_numeric_dtype(temp_service_df_for_idxmax[col_idx]):
                    all_numeric_transform = False; break
            
            if all_numeric_transform and not temp_service_df_for_idxmax.empty and state.max_spend_categories is not None:
                if not temp_service_df_for_idxmax.apply(lambda row: (row == 0).all(), axis=1).all():
                    max_spend_series = temp_service_df_for_idxmax.idxmax(axis=1)
                    X_df["max_spend_category"] = max_spend_series.astype('category').cat.set_categories(state.max_spend_categories).cat.codes
                else: # All rows are all zeros
                     X_df["max_spend_category"] = -1 # Or some other default code
            elif "max_spend_category" not in X_df.columns: 
                X_df["max_spend_category"] = -1 

        if 'Age' in X_df.columns:
            age_fill_value = state.age_median if pd.notna(state.age_median) else (X_df['Age'].median() if pd.notna(X_df['Age'].median()) else 30) 
            X_df['Age'] = X_df['Age'].fillna(age_fill_value)

        if 'Cabin' in X_df.columns:
            cabin_parts = X_df["Cabin"].str.split("/", expand=True)
            if cabin_parts.shape[1] == 3:
                cabin_parts.columns = ["deck_raw", "cabin_num_raw", "side_raw"]
                if state.deck_categories is not None:
                    X_df["deck"] = cabin_parts["deck_raw"].astype('category').cat.set_categories(state.deck_categories).cat.codes
                elif "deck" not in X_df.columns : X_df["deck"] = -1
                
                X_df["cabin_num"] = pd.to_numeric(cabin_parts["cabin_num_raw"], errors="coerce") # KNNImputer handles NaNs later

                if state.side_categories is not None:
                    X_df["side"] = cabin_parts["side_raw"].astype('category').cat.set_categories(state.side_categories).cat.codes
                elif "side" not in X_df.columns: X_df["side"] = -1
            X_df = X_df.drop(columns=["Cabin"], errors='ignore')
        
        X_df = X_df.drop(columns=['Name', 'Ticket'], errors='ignore') # Name handled by NameWordFeatures

        return RawTransformerOutput(X=X_df, y=y_series)

    def _transform_stream(self, data: RawTransformerInput, state: RawTransformerState) -> Iterator[Union[StreamMessage, RawTransformerOutput]]:
        yield StreamMessage(step_name=self.name, status="in_progress", message="Starting raw transformations.", progress=0.0)
        transformed_output = self._transform(data, state)
        yield StreamMessage(step_name=self.name, status="in_progress", message="Raw transformations applied.", progress=1.0)
        yield transformed_output
    
    def _get_state_model(self) -> type[BaseModel]: return RawTransformerState


# 2. KNN Imputer Component
class KNNImputerState(FittedState):
    imputer: Optional[KNNImputer] = None 
    columns: List[str] = Field(default_factory=list) 
    class Config: arbitrary_types_allowed = True

class KNNImputerComponent(ETLRunnable[BaseETLData, TransformedData, KNNImputerState]):
    def __init__(self, n_neighbors: int = 9): 
        super().__init__(name="KNNImputerComponent")
        self.n_neighbors = n_neighbors

    def _fit(self, data: BaseETLData) -> KNNImputerState:
        # Select only numeric columns for imputation
        X_numeric = data.X.select_dtypes(include=np.number) 
        if X_numeric.empty:
            print(f"Warning: No numeric columns found for {self.name} fitting. Imputer will not be active.")
            return KNNImputerState(imputer=None, columns=[])
        
        # Check if there are any NaNs to impute in the numeric columns
        if X_numeric.isnull().sum().sum() == 0:
            print(f"Warning: No NaNs found in numeric columns for {self.name} fitting. Imputer will be fitted but might not change data if no NaNs at transform time.")
            # Fit the imputer anyway, as transform might encounter NaNs
        
        imputer_instance = KNNImputer(n_neighbors=self.n_neighbors)
        imputer_instance.fit(X_numeric) # Fit only on numeric columns
        return KNNImputerState(imputer=imputer_instance, columns=list(X_numeric.columns))

    def _transform(self, data: BaseETLData, state: KNNImputerState) -> TransformedData:
        X_transformed = data.X.copy()
        y_series = data.y.copy() if data.y is not None else None

        if state.imputer is None or not state.columns: 
            return TransformedData(X=X_transformed, y=y_series)
        
        # Select only the numeric columns the imputer was fitted on that are present in the current X
        cols_to_impute_present_numeric = [col for col in state.columns if col in X_transformed.columns and pd.api.types.is_numeric_dtype(X_transformed[col])]
        
        if not cols_to_impute_present_numeric:
            return TransformedData(X=X_transformed, y=y_series)

        X_numeric_subset = X_transformed[cols_to_impute_present_numeric]
        if X_numeric_subset.isnull().sum().sum() == 0: # No NaNs in the relevant subset
            return TransformedData(X=X_transformed, y=y_series)
            
        imputed_array = state.imputer.transform(X_numeric_subset)
        X_imputed_df = pd.DataFrame(imputed_array, columns=X_numeric_subset.columns, index=X_numeric_subset.index)
        
        for col in X_imputed_df.columns: # Update the original DataFrame
            X_transformed[col] = X_imputed_df[col]
            
        return TransformedData(X=X_transformed, y=y_series)

    def _transform_stream(self, data: BaseETLData, state: KNNImputerState) -> Iterator[Union[StreamMessage, TransformedData]]:
        yield StreamMessage(step_name=self.name, status="in_progress", message="Starting KNN imputation.", progress=0.0)
        if state.imputer is None or not state.columns:
            yield StreamMessage(step_name=self.name, status="warning", message="Imputer not active or no columns. Skipping.", progress=0.5)
            yield TransformedData(X=data.X.copy(), y=data.y.copy() if data.y is not None else None); return
        
        transformed_output = self._transform(data,state)
        yield StreamMessage(step_name=self.name, status="in_progress", message="KNN imputation applied if necessary.", progress=1.0)
        yield transformed_output

    def _get_state_model(self) -> type[BaseModel]: return KNNImputerState

# 3. NameWordFeatures Component
class NameWordFeaturesInput(BaseETLData): # Requires y for fitting
    X: pd.DataFrame
    y: pd.Series 

class NameWordFeaturesState(FittedState):
    word_stats: Dict[str, Dict[str, float]]
    name_column: str
    min_count: int

class NameWordFeaturesOutput(TransformedData): pass

class NameWordFeatures(ETLRunnable[NameWordFeaturesInput, NameWordFeaturesOutput, NameWordFeaturesState]):
    def __init__(self, name_column: str = "Name", min_count: int = 5):
        super().__init__(name="NameWordFeatures")
        self.name_column = name_column
        self.min_count = min_count

    def _tokenize(self, text: Any) -> List[str]:
        return re.findall(r"\b\w+\b", str(text).lower())

    def _fit(self, data: NameWordFeaturesInput) -> NameWordFeaturesState:
        if self.name_column not in data.X.columns:
            print(f"Warning: Name column '{self.name_column}' not found in X_train for {self.name} fitting. State will be empty for word_stats.")
            return NameWordFeaturesState(word_stats={}, name_column=self.name_column, min_count=self.min_count)

        X_train, y_train = data.X, data.y # y_train is guaranteed by NameWordFeaturesInput
        word_counts = defaultdict(lambda: [0, 0]) 
        y_train_numeric = y_train.astype(int) # Ensure y is 0/1 
        total_0 = sum(y_train_numeric == 0)
        total_1 = sum(y_train_numeric == 1)

        for name_val, label in zip(X_train[self.name_column], y_train_numeric):
            tokens = self._tokenize(name_val)
            for token in set(tokens): 
                word_counts[token][label] += 1 
        fitted_word_stats = {}
        for word, (c0, c1) in word_counts.items():
            total_occurrences = c0 + c1
            if total_occurrences < self.min_count: continue
            class1_rate = c1 / total_occurrences if total_occurrences > 0 else 0.0
            table = [[c1, c0], [total_1 - c1, total_0 - c0]]
            if (total_1 - c1) < 0 or (total_0 - c0) < 0: continue # Avoid negative counts in Fisher's table
            try:
                _, p_value = fisher_exact(table)
                fitted_word_stats[word] = {"count_0": float(c0), "count_1": float(c1), "class1_rate": float(class1_rate), "p_value": float(p_value)}
            except ValueError: continue # Skip if Fisher's exact test fails (e.g., all zeros in a row/column)
        return NameWordFeaturesState(word_stats=fitted_word_stats, name_column=self.name_column, min_count=self.min_count)

    def _get_best_stats(self, tokens: List[str], current_word_stats: Dict[str, Dict[str, float]]):
        best_pos = {"p": 1.0, "prob": 0.5}; best_neg = {"p": 1.0, "prob": 0.5}
        for token in tokens:
            stats = current_word_stats.get(token)
            if not stats: continue
            if stats["class1_rate"] > 0.5 and stats["p_value"] < best_pos["p"]: best_pos = {"p": stats["p_value"], "prob": stats["class1_rate"]}
            elif stats["class1_rate"] < 0.5 and stats["p_value"] < best_neg["p"]: best_neg = {"p": stats["p_value"], "prob": stats["class1_rate"]}
        return best_pos["p"], best_pos["prob"], best_neg["p"], best_neg["prob"]

    def _transform(self, data: NameWordFeaturesInput, state: NameWordFeaturesState) -> NameWordFeaturesOutput:
        X_df = data.X.copy(); y_series = data.y.copy() if data.y is not None else None # Pass y through
        
        if state.name_column not in X_df.columns or not state.word_stats:
            # If name column is present but no stats, drop it. If not present, do nothing to it.
            if state.name_column in X_df.columns:
                 X_df = X_df.drop(columns=[state.name_column], errors='ignore')
            return NameWordFeaturesOutput(X=X_df, y=y_series)

        pos_p, pos_prob, neg_p, neg_prob_list = [], [], [], [] # Renamed neg_prob to neg_prob_list for clarity
        for name_val in X_df[state.name_column]:
            tokens = self._tokenize(name_val)
            p_pos_val, prob_pos_val, p_neg_val, prob_neg_val = self._get_best_stats(tokens, state.word_stats)
            pos_p.append(p_pos_val); pos_prob.append(prob_pos_val)
            neg_p.append(p_neg_val); neg_prob_list.append(prob_neg_val)
        
        X_df["name_corr_pos_p_value"] = pos_p
        X_df["name_corr_pos_prob"] = pos_prob
        X_df["name_corr_neg_p_value"] = neg_p
        X_df["name_corr_neg_prob"] = neg_prob_list # Use the list
        
        X_df = X_df.drop(columns=[state.name_column]) # Drop original name column
        return NameWordFeaturesOutput(X=X_df, y=y_series)

    def _transform_stream(self, data: NameWordFeaturesInput, state: NameWordFeaturesState) -> Iterator[Union[StreamMessage, NameWordFeaturesOutput]]:
        yield StreamMessage(step_name=self.name, status="in_progress", message="Applying name word features.", progress=0.0)
        output = self._transform(data, state)
        yield StreamMessage(step_name=self.name, status="in_progress", message="Name word features applied.", progress=1.0)
        yield output
    
    def _get_state_model(self) -> type[BaseModel]: return NameWordFeaturesState
    def _get_input_model_fit(self) -> type[BaseModel]: return NameWordFeaturesInput
    def _get_input_model_transform(self) -> type[BaseModel]: return NameWordFeaturesInput


# 4. Feature Generators
class FeatureGeneratorsState(FittedState):
    scaler: Optional[MinMaxScaler] = None 
    ica: Optional[FastICA] = None
    kmeans: Optional[KMeans] = None
    gmm: Optional[GaussianMixture] = None
    feature_columns: List[str] = Field(default_factory=list) 
    class Config: arbitrary_types_allowed = True

class FeatureGenerators(ETLRunnable[BaseETLData, TransformedData, FeatureGeneratorsState]):
    def __init__(self, n_ica_components: Optional[int]=7, n_kmeans_clusters: int=5, n_gmm_components: int=5, random_state=42):
        super().__init__(name="FeatureGenerators")
        self.n_ica = n_ica_components
        self.n_kmeans = n_kmeans_clusters
        self.n_gmm = n_gmm_components
        self.rs = random_state

    def _select_numeric_features(self, X:pd.DataFrame) -> pd.DataFrame:
        # Ensure NaNs are filled before scaling, as MinMaxScaler is sensitive to them.
        # KNNImputer should have run before this, but as a safeguard:
        return X.select_dtypes(include=np.number).fillna(0) 

    def _fit(self, data: BaseETLData) -> FeatureGeneratorsState:
        X_numeric = self._select_numeric_features(data.X)
        if X_numeric.empty: 
            print(f"Warning: No numeric features for {self.name} fitting. State will be minimal.")
            return FeatureGeneratorsState(feature_columns=[])

        scaler = MinMaxScaler().fit(X_numeric)
        # Important: Transform X_numeric with the fitted scaler for subsequent fits (ICA, KMeans, GMM)
        X_scaled_array = scaler.transform(X_numeric) 
        # For clustering, it's often better to use a DataFrame to preserve column names if needed by internals,
        # though sklearn usually works with numpy arrays.
        X_scaled_df_for_clustering = pd.DataFrame(X_scaled_array, columns=X_numeric.columns, index=X_numeric.index)

        
        ica_instance = None
        if self.n_ica and 0 < self.n_ica <= X_scaled_array.shape[1]:
            try: 
                # FastICA's whiten='unit-variance' is similar to StandardScaler, ensure input is appropriate
                # If input is already 0-1 scaled by MinMaxScaler, unit-variance whitening might behave differently.
                # Original etl.py uses MinMaxScaler then ICA.
                ica_instance = FastICA(n_components=self.n_ica, random_state=self.rs, whiten='unit-variance', max_iter=300, tol=0.01).fit(X_scaled_array)
            except Exception as e: print(f"ICA fitting failed for {self.name}: {e}. Skipping ICA."); ica_instance=None
        elif self.n_ica: print(f"Warning for {self.name}: n_ica_components ({self.n_ica}) invalid for data shape {X_scaled_array.shape}. Skipping ICA.")

        n_samples = X_scaled_array.shape[0]
        actual_kmeans_clusters = min(self.n_kmeans, n_samples) if n_samples > 0 else self.n_kmeans
        actual_gmm_components = min(self.n_gmm, n_samples) if n_samples > 0 else self.n_gmm
        
        kmeans_instance, gmm_instance = None, None
        if n_samples == 0:
            print(f"Warning: No samples for clustering in {self.name}.")
        else:
            if actual_kmeans_clusters < self.n_kmeans: print(f"Warning: Reducing kmeans_clusters for {self.name} to {actual_kmeans_clusters} due to low sample count ({n_samples}).")
            if actual_kmeans_clusters > 0: kmeans_instance = KMeans(n_clusters=actual_kmeans_clusters, random_state=self.rs, n_init='auto').fit(X_scaled_df_for_clustering) # Fit on DataFrame
            else: print(f"Warning: Skipping KMeans for {self.name} due to 0 effective clusters.")

            if actual_gmm_components < self.n_gmm: print(f"Warning: Reducing gmm_components for {self.name} to {actual_gmm_components} due to low sample count ({n_samples}).")
            if actual_gmm_components > 0: gmm_instance = GaussianMixture(n_components=actual_gmm_components, random_state=self.rs).fit(X_scaled_df_for_clustering) # Fit on DataFrame
            else: print(f"Warning: Skipping GMM for {self.name} due to 0 effective components.")
            
        return FeatureGeneratorsState(scaler=scaler, ica=ica_instance, kmeans=kmeans_instance, gmm=gmm_instance, feature_columns=list(X_numeric.columns))

    def _transform(self, data: BaseETLData, state: FeatureGeneratorsState) -> TransformedData:
        X_df = data.X.copy() # Operate on a copy
        y_series = data.y.copy() if data.y is not None else None

        if not state.feature_columns or state.scaler is None: 
            return TransformedData(X=X_df, y=y_series)
        
        cols_to_transform = [col for col in state.feature_columns if col in X_df.columns]
        if not cols_to_transform:
            return TransformedData(X=X_df, y=y_series)
            
        X_numeric_subset = self._select_numeric_features(X_df[cols_to_transform]) # fillna(0) happens here
        if X_numeric_subset.empty:
            return TransformedData(X=X_df, y=y_series)

        X_scaled_array = state.scaler.transform(X_numeric_subset)
        # Create a DataFrame from X_scaled_array for clustering, preserving index and column names
        X_scaled_df_for_clustering = pd.DataFrame(X_scaled_array, columns=X_numeric_subset.columns, index=X_numeric_subset.index)

        # ICA components are added to the original X_df, not replacing it
        if state.ica:
            try:
                X_ica_transformed = state.ica.transform(X_scaled_array) # ICA uses the scaled numpy array
                for i in range(X_ica_transformed.shape[1]): X_df[f'ica_{i+1}'] = X_ica_transformed[:,i] 
            except Exception as e: print(f"ICA transform failed for {self.name}: {e}.")
        
        if X_scaled_array.shape[0] > 0: # Ensure there are samples
            if state.kmeans:
                kmeans_labels = state.kmeans.predict(X_scaled_df_for_clustering) 
                X_df['kmeans_cluster'] = kmeans_labels
                kmeans_distances = pairwise_distances(X_scaled_df_for_clustering, state.kmeans.cluster_centers_)
                for i in range(kmeans_distances.shape[1]):
                    X_df[f"kmeans_dist_{i}"] = kmeans_distances[:, i]
            if state.gmm:
                try:
                    gmm_probs = state.gmm.predict_proba(X_scaled_df_for_clustering) 
                    for i in range(gmm_probs.shape[1]): X_df[f'gmm_cluster_prob_{i}'] = gmm_probs[:,i] 
                except Exception as e: print(f"GMM transform failed for {self.name}: {e}.")
        
        # The original numeric columns in X_df are NOT replaced by their scaled versions here.
        # etl.py's transform_with_clustering returns a new DataFrame (X_scaled_df) which includes original scaled columns + new features.
        # To match, we need to decide if X_df should contain scaled versions of original numerics OR original values.
        # Original etl.py: X_train (output of run_custom_etl) has scaled numeric features + new features.
        # So, we should update the original numeric columns in X_df with their scaled versions.
        
        # Update original numeric columns in X_df with their scaled versions
        scaled_numeric_df = pd.DataFrame(X_scaled_array, columns=X_numeric_subset.columns, index=X_numeric_subset.index)
        for col in scaled_numeric_df.columns:
            if col in X_df.columns: # Ensure column still exists
                X_df[col] = scaled_numeric_df[col]
                
        return TransformedData(X=X_df, y=y_series)

    def _transform_stream(self, data: BaseETLData, state: FeatureGeneratorsState) -> Iterator[Union[StreamMessage, TransformedData]]:
        yield StreamMessage(step_name=self.name, status="in_progress", message="Starting feature generation.", progress=0.0)
        output = self._transform(data, state) 
        yield StreamMessage(step_name=self.name, status="in_progress", message="Feature generation applied.", progress=1.0)
        yield output
        
    def _get_state_model(self) -> type[BaseModel]: return FeatureGeneratorsState

# 5. Service Features Component
class ServiceFeaturesComponentInput(BaseETLData): # y is optional for fit
    X: pd.DataFrame
    y: Optional[pd.Series] = None 

class ServiceFeaturesComponentState(FittedState):
    service_cols: List[str]
    quartile_thresholds: Dict[str, Dict[str, float]]

class ServiceFeaturesComponentOutput(TransformedData): pass

class ServiceFeaturesComponent(ETLRunnable[ServiceFeaturesComponentInput, ServiceFeaturesComponentOutput, ServiceFeaturesComponentState]):
    def __init__(self, service_cols: List[str]):
        super().__init__(name="ServiceFeaturesComponent")
        self.service_cols = service_cols

    def _fit(self, data: ServiceFeaturesComponentInput) -> ServiceFeaturesComponentState:
        X_train = data.X # Use X from input for fitting thresholds
        quartile_thresholds: Dict[str, Dict[str, float]] = {}
        # Filter to service columns actually present in X_train
        actual_service_cols_in_X = [col for col in self.service_cols if col in X_train.columns]

        for col in actual_service_cols_in_X:
            if pd.api.types.is_numeric_dtype(X_train[col]):
                # Ensure there are non-NaN values to compute quantiles
                if X_train[col].notna().any():
                    q25, q50, q75 = X_train[col].quantile([0.25, 0.5, 0.75])
                    quartile_thresholds[col] = {"q25": q25, "q50": q50, "q75": q75}
                else:
                    print(f"Warning for {self.name} fit: Service column '{col}' contains all NaNs. Cannot compute quartiles.")
                    # Store default/NaN thresholds or handle as error? For now, skip.
            else:
                print(f"Warning for {self.name} fit: Service column '{col}' is not numeric. Cannot compute quartiles.")
        return ServiceFeaturesComponentState(service_cols=actual_service_cols_in_X, quartile_thresholds=quartile_thresholds)

    def _transform(self, data: ServiceFeaturesComponentInput, state: ServiceFeaturesComponentState) -> ServiceFeaturesComponentOutput:
        X_df = data.X.copy()
        y_series = data.y.copy() if data.y is not None else None

        for col in state.service_cols: # Iterate over service cols known from fit state
            if col not in X_df.columns: # Skip if column is missing in current X_df
                # This can happen if a previous step removed it, though unlikely for service_cols
                print(f"Warning for {self.name} transform: Service column '{col}' not found in X. Skipping feature generation for it.")
                continue
            
            # Ensure column is numeric for comparisons and fillna(0) for has_used logic
            # Original etl.py implies service columns are numeric by this point from KNNImputer
            # and fillna(0) is used for 'has_used' logic.
            numeric_col_data = pd.to_numeric(X_df[col], errors='coerce') # Convert to numeric, non-convertibles become NaN

            X_df[f"has_used_{col}"] = numeric_col_data.fillna(0) > 0 # Based on etl.py's fillna(0) for this check

            if col in state.quartile_thresholds: # Check if thresholds were successfully computed for this column
                # For comparisons, ensure NaNs in numeric_col_data don't cause issues.
                # If a value is NaN, comparisons will be False, which might be desired.
                # Or, fill NaNs before comparison if specific behavior is needed (e.g., fill with 0 or median).
                # etl.py doesn't explicitly fill NaNs in service_cols before quantile comparison in add_service_features,
                # relying on previous imputation.
                X_df[f"{col}_tp25"] = numeric_col_data > state.quartile_thresholds[col]["q25"]
                X_df[f"{col}_tp50"] = numeric_col_data > state.quartile_thresholds[col]["q50"]
                X_df[f"{col}_tp75"] = numeric_col_data > state.quartile_thresholds[col]["q75"]
            else: # Thresholds not available, create dummy columns to maintain structure
                print(f"Warning for {self.name} transform: Quartile thresholds for '{col}' not found in state. Creating dummy boolean columns (all False).")
                X_df[f"{col}_tp25"] = False
                X_df[f"{col}_tp50"] = False
                X_df[f"{col}_tp75"] = False
        
        X_df = X_df.drop(columns=state.service_cols, errors='ignore') # Drop original service columns
        return ServiceFeaturesComponentOutput(X=X_df, y=y_series)

    def _transform_stream(self, data: ServiceFeaturesComponentInput, state: ServiceFeaturesComponentState) -> Iterator[Union[StreamMessage, ServiceFeaturesComponentOutput]]:
        yield StreamMessage(step_name=self.name, status="in_progress", message="Starting service feature engineering.", progress=0.0)
        output = self._transform(data, state) 
        yield StreamMessage(step_name=self.name, status="in_progress", message="Service features engineered and original columns dropped.", progress=1.0)
        yield output

    def _get_state_model(self) -> type[BaseModel]: return ServiceFeaturesComponentState
    def _get_input_model_fit(self) -> type[BaseModel]: return ServiceFeaturesComponentInput


# --- Helper to get raw dataset (Removed placeholder fill) ---
def get_raw_dataset(dataset_name: str, target_column: str, dataset_type: str = "raw", drop_na: bool = False) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    raw_data_target_dir = None 
    try:
        # Try standard Jovyan path first
        base_data_dir_jovyan = Path("/home/jovyan/data")
        # Fallback for local execution if Jovyan path doesn't exist or is not accessible
        base_data_dir_local = Path("./data")

        if base_data_dir_jovyan.exists() and base_data_dir_jovyan.is_dir():
            base_data_dir = base_data_dir_jovyan
        elif base_data_dir_local.exists() and base_data_dir_local.is_dir():
            base_data_dir = base_data_dir_local
            print(f"Using local data directory: {base_data_dir.resolve()}")
        else:
            # If neither exists, default to attempting local, error will be caught by glob if no files
            base_data_dir = base_data_dir_local 
            print(f"Warning: Neither /home/jovyan/data nor ./data found. Attempting with {base_data_dir.resolve()}")


        dataset_specific_base_path = base_data_dir / dataset_name
        raw_data_target_dir = dataset_specific_base_path / dataset_type

        if not raw_data_target_dir.is_dir():
            # This will be caught by FileNotFoundError if glob returns no files
            pass

        csv_files = list(raw_data_target_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {raw_data_target_dir.resolve() if raw_data_target_dir else 'Path not determined'}")

        print(f"Attempting to load from: {csv_files[0]}")
        df = pd.read_csv(csv_files[0])
        print(f"Dataset loaded successfully from {csv_files[0]}. Shape: {df.shape}")

        # DO NOT apply fill_missing_values_placeholder here. Imputation is handled by pipeline components.

        y: Optional[pd.Series] = None; X: pd.DataFrame

        if target_column not in df.columns and target_column != "placeholder":
            print(f"Warning: Target column '{target_column}' not found. Setting y to None.")
            X = df
        elif target_column == "placeholder": 
            X = df
            common_targets = ["Transported", "quality", "target"] 
            for ct in common_targets: # Remove known target names if they appear in test data
                if ct in X.columns and (X[ct].dtype == bool or X[ct].dtype == object): # Check type before dropping
                     X = X.drop(columns=[ct], errors='ignore')
        else:
            y = df[target_column]
            X = df.drop(columns=[target_column], errors='ignore')

        if drop_na: # This is True only for the initial raw load in etl.py's run_custom_etl if specified
            if y is not None:
                valid_indices = y.dropna().index
                y = y.loc[valid_indices].reset_index(drop=True)
                X = X.loc[valid_indices].reset_index(drop=True)
            else: # If y is None but drop_na is True, drop all rows with any NA in X
                X = X.dropna().reset_index(drop=True)
        
        if y is not None and target_column == "Transported": y = y.astype(bool)

        return X, y
    except FileNotFoundError as e:
        error_message = (f"Data loading failed. Dir: '{raw_data_target_dir.resolve() if raw_data_target_dir else 'Path not determined'}'. Error: {e}")
        print(error_message); raise FileNotFoundError(error_message) from e


# --- Main ETL Runner ---
def run_custom_etl_streaming(dataset_name: str, target_column: str, test_split: float = .2) -> dict:
    # Load training data, do not drop NaNs at this stage (drop_na=False). Components will handle.
    X_raw, y_raw = get_raw_dataset(dataset_name, target_column, dataset_type="raw", drop_na=False) 
        
    if X_raw is None: # y_raw can be None for test data loading, but not for training
         raise ValueError(f"Training data X_raw could not be loaded for dataset '{dataset_name}'.")
    if y_raw is None and target_column != "placeholder": # Ensure y_raw is loaded for training
        raise ValueError(f"Target column '{target_column}' not found or y_raw is None for training.")


    initial_X_df = X_raw
    initial_y_series = y_raw if isinstance(y_raw, pd.Series) else pd.Series(y_raw) if y_raw is not None else None
    
    # For train_test_split, ensure y does not contain NaNs if stratifying
    # If y_raw is None (e.g. loading test data for some reason here, though unlikely), skip split
    if initial_y_series is None:
        raise ValueError("Cannot perform train_test_split: y_raw is None.")

    valid_y_indices = initial_y_series.dropna().index
    X_for_split = initial_X_df.loc[valid_y_indices]
    y_for_split = initial_y_series.loc[valid_y_indices]

    if X_for_split.empty or y_for_split.empty:
        raise ValueError("Not enough data after dropping NaNs in target for train_test_split.")

    can_stratify = y_for_split.nunique() > 1 and all(y_for_split.value_counts() >= 2)

    X_train_raw_df, X_val_raw_df, y_train_series, y_val_series = train_test_split(
        X_for_split, y_for_split, 
        test_size=test_split, random_state=42, 
        stratify=y_for_split if can_stratify else None
    )
    X_train_raw_df = X_train_raw_df.reset_index(drop=True)
    X_val_raw_df = X_val_raw_df.reset_index(drop=True)
    y_train_series = y_train_series.reset_index(drop=True)
    y_val_series = y_val_series.reset_index(drop=True)

    X_train_unprocessed_for_return = X_train_raw_df.copy() 

    service_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    raw_transformer = RawTransformer(service_cols=service_cols)
    knn_imputer = KNNImputerComponent(n_neighbors=9) # As per etl.py
    name_feature_processor = NameWordFeatures(name_column='Name', min_count=5) 
    feature_generator = FeatureGenerators(
        n_ica_components=7, 
        n_kmeans_clusters=5, 
        n_gmm_components=5, 
        random_state=42
    )
    service_feature_processor = ServiceFeaturesComponent(service_cols=service_cols)

    # Pipeline order to match etl.py: Raw -> KNNImpute -> Name -> FeatureGen -> ServiceFeat
    fit_pipeline = raw_transformer | knn_imputer | name_feature_processor | feature_generator | service_feature_processor
    
    # For fitting, the chain's _fit method will pass appropriate inputs to each step
    print(X_train_raw_df.columns)
    train_data_for_fit = BaseETLData(X=X_train_raw_df.copy(), y=y_train_series.copy())

    print("--- Fitting pipeline on Training Data ---")
    fit_pipeline.fit(train_data_for_fit)

    train_data_for_transform = BaseETLData(X=X_train_raw_df.copy(), y=y_train_series.copy())
    val_data_for_transform = BaseETLData(X=X_val_raw_df.copy(), y=y_val_series.copy())

    print("\n--- Transforming Training Data (Streaming) ---")
    final_train_data = None
    train_stream = fit_pipeline.transform_stream(train_data_for_transform)
    for update in train_stream:
        if isinstance(update, StreamMessage):
            print(f"STREAM_MSG (Train): [{update.step_name}] Status: {update.status}, Msg: {str(update.message)[:150]}, Progress: {update.progress or 'N/A'}")
        else: final_train_data = update
    if final_train_data is None: raise ValueError("Training data transformation stream did not yield final data.")
    X_train, y_train = final_train_data.X, final_train_data.y

    print("\n--- Transforming Validation Data (Streaming) ---")
    final_val_data = None
    val_stream = fit_pipeline.transform_stream(val_data_for_transform)
    for update in val_stream:
        if isinstance(update, StreamMessage):
            print(f"STREAM_MSG (Val): [{update.step_name}] Status: {update.status}, Msg: {str(update.message)[:150]}, Progress: {update.progress or 'N/A'}")
        else: final_val_data = update
    if final_val_data is None: raise ValueError("Validation data transformation stream did not yield final data.")
    X_val, y_val = final_val_data.X, final_val_data.y

    print("\n--- Loading and Transforming Test Data ---")
    X_test_raw, _ = get_raw_dataset(dataset_name, target_column="placeholder", dataset_type="test", drop_na=False)
    passenger_ids: Optional[pd.Series] = None
    X_test = None

    if X_test_raw is not None:
        if "PassengerId" in X_test_raw.columns: # PassengerId is kept in X_test_raw by get_raw_dataset
            passenger_ids = X_test_raw["PassengerId"].copy()
            # RawTransformer will drop PassengerId from its input X
        else:
            print("Warning: 'PassengerId' column not found in test data.")
        
        dummy_y_for_test = pd.Series([None] * len(X_test_raw), index=X_test_raw.index, dtype=object) if X_test_raw is not None else None
        test_data_for_transform = BaseETLData(X=X_test_raw.copy(), y=dummy_y_for_test) # Pass full X_test_raw
        
        final_test_data = None
        test_stream = fit_pipeline.transform_stream(test_data_for_transform)
        for update in test_stream:
            if isinstance(update, StreamMessage):
                print(f"STREAM_MSG (Test): [{update.step_name}] Status: {update.status}, Msg: {str(update.message)[:150]}, Progress: {update.progress or 'N/A'}")
            else: final_test_data = update
        
        if final_test_data is None:
            print("Warning: Test data transformation stream did not yield final data.")
        else:
            X_test = final_test_data.X
            # Align X_test columns with X_train
            train_cols = X_train.columns
            test_cols = X_test.columns
            missing_in_test = list(set(train_cols) - set(test_cols))
            extra_in_test = list(set(test_cols) - set(train_cols))

            for col in missing_in_test: X_test[col] = 0 
            if extra_in_test: X_test = X_test.drop(columns=extra_in_test)
            X_test = X_test[train_cols] 
    else:
        print("Warning: Test data (X_test_raw) could not be loaded.")

    print("\n--- ETL Process Complete ---")
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    if X_test is not None: print(f"X_test shape: {X_test.shape}")
    # print(f"X_train columns: {X_train.columns.tolist()}") 

    etl_description_str = "ETL Pipeline (Chain): " + " -> ".join([step.name for step in fit_pipeline.steps])
    results_dict = {
        "X_train_unprocessed": X_train_unprocessed_for_return,
        "X_train": X_train, "y_train": y_train, 
        "X_val": X_val, "y_val": y_val,
        "etl_description": etl_description_str,
        "fitted_pipeline_chain_state": fit_pipeline.fitted_state.dict() if fit_pipeline.fitted_state else None,
    }
    
    if X_test is not None: 
        results_dict["X_test"] = X_test
        print(f"X_test shape: {X_test.shape}")
    else:
        print("X_test is None. Cannot generate submission.")
    if passenger_ids is not None: results_dict["passenger_ids"] = passenger_ids


    if X_train is None or y_train is None or X_val is None or y_val is None:
        raise ValueError("ETL result is missing training or validation data (X_train, y_train, X_val, y_val).")
        
    return results_dict

# Removed the if __name__ == '__main__': block

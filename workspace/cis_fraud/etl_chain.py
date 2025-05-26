import pandas as pd
from pydantic import BaseModel, Field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union, Iterator, Tuple
from sklearn.model_selection import train_test_split
# from sklearn.impute import KNNImputer # Commented out as per request
# from sklearn.preprocessing import MinMaxScaler # Commented out as per request
# from sklearn.decomposition import FastICA # Commented out as per request
# from sklearn.cluster import KMeans # Commented out as per request
# from sklearn.mixture import GaussianMixture # Commented out as per request
# from sklearn.metrics import pairwise_distances # Commented out as per request
import numpy as np
import time
import os
import re
# from collections import defaultdict # Commented out as per request
# from scipy.stats import fisher_exact # Commented out as per request

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

            # Simplified input data for the current setup
            step_input_data = BaseETLData(X=current_data_x, y=current_data_y)

            print(f"Fitting step in chain: {step.name}")
            step.fit(step_input_data)
            if step.fitted_state:
                all_step_states_dict[f"{step.name}_{i}_state_dict"] = step.fitted_state.dict()

            if step.fitted_state is not None:
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

class RawTransformerState(FittedState):
    numeric_medians: Dict[str, float] = Field(default_factory=dict)
    categorical_modes: Dict[str, Any] = Field(default_factory=dict)
    categorical_column_categories: Dict[str, List[Any]] = Field(default_factory=dict)
    all_columns_fitted: List[str] = Field(default_factory=list)


class RawTransformerInput(BaseETLData): pass
class RawTransformerOutput(TransformedData): pass

class RawTransformer(ETLRunnable[RawTransformerInput, RawTransformerOutput, RawTransformerState]):
    def __init__(self):
        super().__init__(name="RawTransformer")

    def _fit(self, data: RawTransformerInput) -> RawTransformerState:
        X = data.X.copy()
        numeric_medians: Dict[str, float] = {}
        categorical_modes: Dict[str, Any] = {}
        categorical_column_categories: Dict[str, List[Any]] = {}
        all_columns_fitted = list(X.columns)

        cols_to_drop_fit = ["TransactionID", "TransactionDT"]
        X_fit_processed = X.drop(columns=[col for col in cols_to_drop_fit if col in X.columns], errors='ignore')

        for col in X_fit_processed.columns:
            if pd.api.types.is_numeric_dtype(X_fit_processed[col]):
                numeric_medians[col] = X_fit_processed[col].median()
            else:
                mode_val = X_fit_processed[col].mode()
                categorical_modes[col] = mode_val[0] if not mode_val.empty else "missing"
                try:
                    col_as_str = X_fit_processed[col].fillna("NaN_placeholder_fit").astype(str)
                    unique_cats = list(col_as_str.astype('category').cat.categories)
                    categorical_column_categories[col] = unique_cats
                except Exception as e:
                    print(f"Warning: Could not determine categories for column {col} during fit: {e}. It will be treated as object.")
                    categorical_column_categories[col] = [str(categorical_modes[col])]


        return RawTransformerState(
            numeric_medians=numeric_medians,
            categorical_modes=categorical_modes,
            categorical_column_categories=categorical_column_categories,
            all_columns_fitted=all_columns_fitted
        )

    def _transform(self, data: RawTransformerInput, state: RawTransformerState) -> RawTransformerOutput:
        X_df = data.X.copy()
        y_series = data.y.copy() if data.y is not None else None

        cols_to_drop_transform = ["TransactionID", "TransactionDT"]
        X_df_processed = X_df.drop(columns=[col for col in cols_to_drop_transform if col in X_df.columns], errors='ignore')

        for col in X_df_processed.columns:
            if col in state.numeric_medians:
                X_df_processed[col] = pd.to_numeric(X_df_processed[col], errors='coerce')
                X_df_processed[col] = X_df_processed[col].fillna(state.numeric_medians[col])
            elif col in state.categorical_modes:
                X_df_processed[col] = X_df_processed[col].fillna(state.categorical_modes[col])
                if col in state.categorical_column_categories:
                    try:
                        col_as_str_transform = X_df_processed[col].fillna("NaN_placeholder_transform").astype(str)
                        # Ensure all categories in state are strings for pd.Categorical
                        categories_as_str = [str(cat) for cat in state.categorical_column_categories[col]]
                        X_df_processed[col] = pd.Categorical(col_as_str_transform, categories=categories_as_str).codes
                    except Exception as e:
                        print(f"Warning: Could not apply categories for column {col} during transform: {e}. Encoding based on mode or default.")
                        mode_category_code = 0
                        try:
                            # Ensure mode is string for index lookup
                            mode_category_code = categories_as_str.index(str(state.categorical_modes[col]))
                        except ValueError: pass
                        except Exception: pass
                        X_df_processed[col] = X_df_processed[col].apply(lambda x: mode_category_code if str(x) == str(state.categorical_modes[col]) else -1)
                else:
                    X_df_processed[col] = X_df_processed[col].astype(str).astype('category').cat.codes
            else:
                if pd.api.types.is_numeric_dtype(X_df_processed[col]):
                    X_df_processed[col] = X_df_processed[col].fillna(0)
                else:
                    X_df_processed[col] = X_df_processed[col].fillna("missing_new_col").astype(str).astype('category').cat.codes
        return RawTransformerOutput(X=X_df_processed, y=y_series)

    def _transform_stream(self, data: RawTransformerInput, state: RawTransformerState) -> Iterator[Union[StreamMessage, RawTransformerOutput]]:
        yield StreamMessage(step_name=self.name, status="in_progress", message="Starting raw transformations and fillna.", progress=0.0)
        transformed_output = self._transform(data, state)
        yield StreamMessage(step_name=self.name, status="in_progress", message="Raw transformations and fillna applied.", progress=1.0)
        yield transformed_output

    def _get_state_model(self) -> type[BaseModel]: return RawTransformerState

def get_raw_dataset_generator(
    output_chunk_dir_path: Union[str, Path], # Path to the directory with merged chunks
    file_format: str,                       # "parquet" or "csv"
    target_column: str,
    drop_na: bool = False
    # Removed dataset_name, dataset_type, and chunksize (for reading) as they are not directly applicable here
) -> Iterator[Tuple[pd.DataFrame, Optional[pd.Series]]]:
    """
    Generator function to load and process dataset directly from pre-merged chunk files.
    """
    output_chunk_dir_path = Path(output_chunk_dir_path) # Ensure it's a Path object

    if not output_chunk_dir_path.is_dir():
        error_message = f"Error: Merged chunks directory not found at {output_chunk_dir_path}. Please check the path."
        print(error_message)
        raise FileNotFoundError(error_message)

    print(f"Looking for merged chunk files in: {output_chunk_dir_path}")
    print(f"Expected file format: {file_format}")

    # Get a list of all your chunk files
    try:
        if file_format.lower() == "parquet":
            chunk_files = sorted([
                os.path.join(output_chunk_dir_path, f)
                for f in os.listdir(output_chunk_dir_path)
                if f.startswith("merged_chunk_") and f.endswith('.parquet')
            ])
        elif file_format.lower() == "csv":
            chunk_files = sorted([
                os.path.join(output_chunk_dir_path, f)
                for f in os.listdir(output_chunk_dir_path)
                if f.startswith("merged_chunk_") and f.endswith('.csv')
            ])
        else:
            raise ValueError(f"Unsupported file_format: {file_format}. Must be 'parquet' or 'csv'.")
            
    except FileNotFoundError: # This might occur if os.listdir fails on a non-existent path, though is_dir check is above
        print(f"Error: Directory not found at {output_chunk_dir_path} when listing files. Please check the path.")
        chunk_files = [] # Should not be reached if initial check passes
    
    if not chunk_files:
        print(f"No '{file_format}' chunk files starting with 'merged_chunk_' found in {output_chunk_dir_path}.")
        return # Stop iteration if no files found

    print(f"Found {len(chunk_files)} chunk files to process.")

    for i, chunk_file_path in enumerate(chunk_files):
        print(f"Processing chunk file {i+1}/{len(chunk_files)}: {chunk_file_path}")
        try:
            if file_format.lower() == "parquet":
                df_chunk = pd.read_parquet(chunk_file_path)
            else: # CSV
                df_chunk = pd.read_csv(chunk_file_path)
            
            print(f"  Loaded chunk shape: {df_chunk.shape}")

            y_chunk: Optional[pd.Series] = None
            X_chunk: pd.DataFrame

            if target_column != "placeholder" and target_column in df_chunk.columns:
                y_chunk = df_chunk[target_column]
                X_chunk = df_chunk.drop(columns=[target_column], errors='ignore')
            elif target_column == "placeholder": # For test data where target is not present
                X_chunk = df_chunk
            else:
                print(f"  Warning: Target column '{target_column}' not found in chunk file {chunk_file_path}. Yielding X only.")
                X_chunk = df_chunk

            if drop_na:
                if y_chunk is not None:
                    # Align indices before dropping NaNs based on y_chunk
                    X_chunk = X_chunk.loc[y_chunk.index] 
                    valid_indices = y_chunk.dropna().index
                    y_chunk = y_chunk.loc[valid_indices].reset_index(drop=True)
                    X_chunk = X_chunk.loc[valid_indices].reset_index(drop=True)
                else: # If y_chunk is None (e.g. placeholder for test data), drop based on X_chunk
                     X_chunk = X_chunk.dropna().reset_index(drop=True)
            
            if y_chunk is not None and target_column == "isFraud": # Specific to this dataset's target
                y_chunk = y_chunk.astype(int)

            if not X_chunk.empty:
                yield X_chunk, y_chunk
            else:
                print(f"  Chunk file {chunk_file_path} resulted in an empty DataFrame after processing, skipping yield.")

        except Exception as e:
            print(f"  Error processing chunk file {chunk_file_path}: {e}")
            # Decide if you want to skip the chunk or raise the error
            # continue # to skip
            raise # to stop

# --- Main ETL Runner ---
def run_custom_etl_streaming(dataset_name: str, target_column: str, test_split: float = .2, chunksize_load: int = 50000) -> dict:
    if dataset_name != "cis_fraud":
        print(f"Warning: This ETL chain is primarily configured for 'cis_fraud'. Results for '{dataset_name}' may vary.")

    # --- Load and prepare data for train_test_split ---
    # For train_test_split, we still need to load a representation of the full dataset
    # or at least enough to make a meaningful split.
    # This is a challenge with true chunk-wise loading from the start if the dataset is huge.
    # Option 1: Load all data first for split, then re-stream for ETL (current approach below).
    # Option 2: Stream once to get indices, then stream again for split (complex).
    # Option 3: Approximate split or use a library that supports out-of-core train_test_split.

    # For now, let's load all data for the split, then the ETL will use the generator.
    # This means get_raw_dataset_generator won't be used for the initial load before split.
    # We need a non-generator version for this initial load or collect all chunks.
    
    print("Initial load for train_test_split (loading full data for this step)...")
    all_X_chunks = []
    all_y_chunks = []
    # Use the generator to load all data into memory for the split
    # This negates some benefits of chunking if the full dataset is too large for memory here.
    # A more advanced setup would handle splitting without loading all data.
    temp_data_gen = get_raw_dataset_generator(dataset_name, target_column, dataset_type="raw", drop_na=False, chunksize=chunksize_load)
    for X_c, y_c in temp_data_gen:
        all_X_chunks.append(X_c)
        if y_c is not None:
            all_y_chunks.append(y_c)

    if not all_X_chunks:
        raise ValueError(f"No data loaded by generator for dataset '{dataset_name}' to perform train/test split.")

    initial_X_df = pd.concat(all_X_chunks).reset_index(drop=True)
    initial_y_series = pd.concat(all_y_chunks).reset_index(drop=True) if all_y_chunks else None
    
    print(f"Full data loaded for split: X shape {initial_X_df.shape}, y shape {initial_y_series.shape if initial_y_series is not None else 'N/A'}")


    if initial_y_series is None and target_column != "placeholder":
        raise ValueError(f"Target column '{target_column}' resulted in None for y_raw after loading all chunks.")

    # Proceed with train_test_split as before
    if initial_y_series is None: # Should only happen if target_column is "placeholder"
         X_train_raw_df, X_val_raw_df = train_test_split(initial_X_df, test_size=test_split, random_state=42)
         y_train_series, y_val_series = None, None # No target to split
    else:
        valid_y_indices = initial_y_series.dropna().index
        X_for_split = initial_X_df.loc[valid_y_indices]
        y_for_split = initial_y_series.loc[valid_y_indices]

        if X_for_split.empty or y_for_split.empty:
            raise ValueError("Not enough data after dropping NaNs in target for train_test_split.")

        can_stratify = y_for_split.nunique() > 1 and all(y_for_split.value_counts(dropna=False) >= 2)
        X_train_raw_df, X_val_raw_df, y_train_series, y_val_series = train_test_split(
            X_for_split, y_for_split,
            test_size=test_split, random_state=42,
            stratify=y_for_split if can_stratify else None
        )

    X_train_raw_df = X_train_raw_df.reset_index(drop=True)
    X_val_raw_df = X_val_raw_df.reset_index(drop=True)
    if y_train_series is not None: y_train_series = y_train_series.reset_index(drop=True)
    if y_val_series is not None: y_val_series = y_val_series.reset_index(drop=True)


    X_train_unprocessed_for_return = X_train_raw_df.copy()

    raw_transformer = RawTransformer()
    fit_pipeline = raw_transformer

    # --- Fit the pipeline ---
    # The fit method of RawTransformer expects a full DataFrame to learn medians, modes, categories.
    # If we were to fit chunk-wise, RawTransformer._fit would need to accumulate stats over chunks.
    # For now, fit on the X_train_raw_df (which is in memory after the split).
    print("--- Fitting pipeline on Training Data (using in-memory X_train_raw_df for fit) ---")
    # Ensure y_train_series is not None if RawTransformer's fit expects it (it doesn't currently)
    train_data_for_fit = BaseETLData(X=X_train_raw_df.copy(), y=y_train_series.copy() if y_train_series is not None else None)
    fit_pipeline.fit(train_data_for_fit)

    # --- Transform Data in Chunks (Illustrative - how you might use the generator concept) ---
    # The current ETLChain.transform_stream expects a single BaseETLData input, not a generator.
    # To use get_raw_dataset_generator directly for transformation by the pipeline,
    # ETLChain and its components would need to be adapted to process an iterator of chunks.

    # For this example, we'll demonstrate transforming the already split (in-memory) data
    # using the fitted pipeline. If these were too large, you'd stream them from disk.

    print("\n--- Transforming Training Data (using in-memory X_train_raw_df) ---")
    # This transform will use the already loaded X_train_raw_df, not re-stream it.
    train_data_for_transform = BaseETLData(X=X_train_raw_df.copy(), y=y_train_series.copy() if y_train_series is not None else None)
    final_train_data = None
    train_stream = fit_pipeline.transform_stream(train_data_for_transform)
    for update in train_stream:
        if isinstance(update, StreamMessage):
            print(f"STREAM_MSG (Train): [{update.step_name}] Status: {update.status}, Msg: {str(update.message)[:150]}, Progress: {update.progress or 'N/A'}")
        else: final_train_data = update
    if final_train_data is None: raise ValueError("Training data transformation stream did not yield final data.")
    X_train, y_train = final_train_data.X, final_train_data.y

    print("\n--- Transforming Validation Data (using in-memory X_val_raw_df) ---")
    val_data_for_transform = BaseETLData(X=X_val_raw_df.copy(), y=y_val_series.copy() if y_val_series is not None else None)
    final_val_data = None
    val_stream = fit_pipeline.transform_stream(val_data_for_transform)
    for update in val_stream:
        if isinstance(update, StreamMessage):
            print(f"STREAM_MSG (Val): [{update.step_name}] Status: {update.status}, Msg: {str(update.message)[:150]}, Progress: {update.progress or 'N/A'}")
        else: final_val_data = update
    if final_val_data is None: raise ValueError("Validation data transformation stream did not yield final data.")
    X_val, y_val = final_val_data.X, final_val_data.y


    print("\n--- Loading and Transforming Test Data (Streaming from disk) ---")
    transaction_ids_test: Optional[pd.Series] = None
    X_test_transformed_chunks = []

    # Use the generator for test data loading
    test_data_generator = get_raw_dataset_generator(
        dataset_name,
        target_column="placeholder", # No target for test data
        dataset_type="test",
        drop_na=False, # Transformers handle NaNs
        chunksize=chunksize_load
    )

    first_test_chunk = True
    for X_test_chunk_raw, _ in test_data_generator: # y_chunk will be None for placeholder
        if X_test_chunk_raw.empty:
            continue

        if first_test_chunk and "TransactionID" in X_test_chunk_raw.columns:
            # Assuming TransactionID is consistent across chunks if taken from the first
            # A more robust way would be to collect all TransactionIDs if needed for submission
            transaction_ids_test = X_test_chunk_raw["TransactionID"].copy()
            first_test_chunk = False
        elif "TransactionID" in X_test_chunk_raw.columns and transaction_ids_test is not None:
             # Concatenate if TransactionIDs are needed from all chunks
             transaction_ids_test = pd.concat([transaction_ids_test, X_test_chunk_raw["TransactionID"]]).drop_duplicates().reset_index(drop=True)


        dummy_y_for_chunk = pd.Series([None] * len(X_test_chunk_raw), index=X_test_chunk_raw.index, dtype=object)
        test_chunk_for_transform = BaseETLData(X=X_test_chunk_raw.copy(), y=dummy_y_for_chunk)

        final_test_chunk_transformed = None
        test_chunk_stream = fit_pipeline.transform_stream(test_chunk_for_transform)
        for update in test_chunk_stream:
            if isinstance(update, StreamMessage):
                # Less verbose logging for per-chunk transform of test data
                if update.status in ["starting", "completed"]: # Log start/end of chunk transform
                     print(f"STREAM_MSG (Test Chunk): [{update.step_name}] Status: {update.status}, Msg: {str(update.message)[:100]}")
            else:
                final_test_chunk_transformed = update
        
        if final_test_chunk_transformed is not None:
            X_test_transformed_chunks.append(final_test_chunk_transformed.X)
        else:
            print("Warning: A test data chunk transformation stream did not yield final data.")

    X_test = None
    if X_test_transformed_chunks:
        X_test = pd.concat(X_test_transformed_chunks).reset_index(drop=True)
        # Align X_test columns with X_train
        train_cols = X_train.columns
        test_cols = X_test.columns
        missing_in_test = list(set(train_cols) - set(test_cols))
        extra_in_test = list(set(test_cols) - set(train_cols))

        for col_miss in missing_in_test: X_test[col_miss] = 0
        if extra_in_test: X_test = X_test.drop(columns=extra_in_test, errors='ignore')
        if not X_test.empty: # Ensure X_test is not empty before reordering
            X_test = X_test[train_cols]
    else:
        print("Warning: No test data chunks were processed or all were empty.")


    print("\n--- ETL Process Complete ---")
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    if X_test is not None: print(f"X_test shape: {X_test.shape}")

    etl_description_str = "ETL Pipeline (Chain): " + " -> ".join([step.name for step in fit_pipeline.steps])
    results_dict = {
        "X_train_unprocessed": X_train_unprocessed_for_return,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "etl_description": etl_description_str,
        "fitted_pipeline_chain_state": fit_pipeline.fitted_state.dict() if fit_pipeline.fitted_state else None,
        "etl_version": "cis_fraud_v2_raw_generator"
    }

    if X_test is not None:
        results_dict["X_test"] = X_test
    else:
        print("X_test is None.")
    if transaction_ids_test is not None:
        results_dict["transaction_ids_test"] = transaction_ids_test


    if X_train is None or y_train is None or X_val is None or y_val is None:
        # y_train and y_val could be None if target_column was "placeholder" for the initial load.
        # This check should be more nuanced if placeholder target is a valid scenario for training.
        if target_column != "placeholder":
            raise ValueError("ETL result is missing training or validation data (X_train, y_train, X_val, y_val).")

    return results_dict



# doesnt work to yield for test rn, see below for updated version
def run_custom_etl_generator(
    output_chunk_dir_path: Union[str, Path],
    file_format: str,
    target_column: str,
    num_chunks_to_fit_on: int = 1,
    num_processed_chunks_to_yield: Union[int, str] = "all",
    drop_na_during_load: bool = False
) -> Iterator[Tuple[pd.DataFrame, Optional[pd.Series]]]:
    """
    Generator function that loads PRE-MERGED data in chunks, 
    fits an ETL transformer (RawTransformer) on a specified number of initial chunks, 
    and then yields processed data chunks.
    """
    print(f"Initializing ETL generator for pre-merged chunks from: {output_chunk_dir_path}")
    print(f"Transformer will be fit on {num_chunks_to_fit_on} initial chunks from this directory.")
    # chunksize_load is no longer needed here as get_raw_dataset_generator reads full chunk files.

    raw_transformer = RawTransformer()

    # --- Fit the RawTransformer ---
    print("Step 1: Loading initial chunks from pre-merged files for fitting the transformer...")
    initial_raw_chunks_for_fit_X = []
    initial_raw_chunks_for_fit_y = []
    
    fit_data_gen = get_raw_dataset_generator(
        output_chunk_dir_path=output_chunk_dir_path,
        file_format=file_format,
        target_column=target_column,
        drop_na=drop_na_during_load # Typically False, as transformer handles NaNs
    )

    for i, (X_raw_chunk, y_raw_chunk) in enumerate(fit_data_gen):
        if i < num_chunks_to_fit_on:
            if X_raw_chunk is not None and not X_raw_chunk.empty:
                initial_raw_chunks_for_fit_X.append(X_raw_chunk)
                if y_raw_chunk is not None:
                    initial_raw_chunks_for_fit_y.append(y_raw_chunk)
            if (i + 1) % 2 == 0 or (i+1) == num_chunks_to_fit_on: 
                print(f"  Loaded {i+1}/{num_chunks_to_fit_on} chunks for fitting...")
        else:
            print(f"  Finished loading {num_chunks_to_fit_on} chunks for fitting.")
            break
    
    del fit_data_gen

    if not initial_raw_chunks_for_fit_X:
        raise ValueError("No data chunks were loaded for fitting the transformer. Check data source or parameters.")

    X_for_fit = pd.concat(initial_raw_chunks_for_fit_X).reset_index(drop=True)
    y_for_fit = pd.concat(initial_raw_chunks_for_fit_y).reset_index(drop=True) if initial_raw_chunks_for_fit_y and any(c is not None for c in initial_raw_chunks_for_fit_y) else None
    
    del initial_raw_chunks_for_fit_X, initial_raw_chunks_for_fit_y

    print(f"Fitting RawTransformer on concatenated data of shape: X={X_for_fit.shape}, y={y_for_fit.shape if y_for_fit is not None else 'N/A'}")
    fit_input_data = BaseETLData(X=X_for_fit, y=y_for_fit)
    raw_transformer.fit(fit_input_data)
    print("RawTransformer fitting complete.")
    del X_for_fit, y_for_fit, fit_input_data

    # --- Process and Yield Chunks ---
    print("\nStep 2: Streaming pre-merged chunks, transforming, and yielding processed chunks...")
    main_data_gen = get_raw_dataset_generator(
        output_chunk_dir_path=output_chunk_dir_path,
        file_format=file_format,
        target_column=target_column,
        drop_na=drop_na_during_load
    )

    chunks_yielded_count = 0
    for i, (X_raw_chunk, y_raw_chunk) in enumerate(main_data_gen):
        if X_raw_chunk is None or X_raw_chunk.empty:
            print(f"  Raw chunk file {i+1} is None or empty, skipping.")
            continue
        
        print(f"  Transforming chunk from file {i+1} (shape: {X_raw_chunk.shape})...")
        
        if y_raw_chunk is not None and not X_raw_chunk.index.equals(y_raw_chunk.index):
            y_raw_chunk = y_raw_chunk.set_axis(X_raw_chunk.index)

        transform_input_data = BaseETLData(X=X_raw_chunk.copy(), y=y_raw_chunk.copy() if y_raw_chunk is not None else None)
        
        processed_chunk_data = None
        transform_gen = raw_transformer.transform_stream(transform_input_data)
        for item in transform_gen:
            if isinstance(item, TransformedData):
                processed_chunk_data = item
                break
        
        if processed_chunk_data and processed_chunk_data.X is not None:
            processed_X_chunk = processed_chunk_data.X
            processed_y_chunk = processed_chunk_data.y

            if processed_y_chunk is not None and not processed_X_chunk.index.equals(processed_y_chunk.index):
                 processed_y_chunk = processed_y_chunk.set_axis(processed_X_chunk.index)

            print(f"    Yielding processed chunk {chunks_yielded_count + 1} (X shape: {processed_X_chunk.shape}, y shape: {processed_y_chunk.shape if processed_y_chunk is not None else 'N/A'})")
            yield processed_X_chunk, processed_y_chunk
            chunks_yielded_count += 1

            if num_processed_chunks_to_yield != "all" and chunks_yielded_count >= num_processed_chunks_to_yield:
                print(f"  Reached yield limit of {num_processed_chunks_to_yield} processed chunks.")
                break
        else:
            print(f"  Skipping yield for raw chunk file {i+1} as transformation did not produce data.")
            
    del main_data_gen
    print(f"\nFinished yielding processed chunks. Total yielded: {chunks_yielded_count}")



def run_custom_etl_generator(
    # Data source parameters (provide one set)
    output_chunk_dir_path: Union[str, Path] = None, # For pre-merged training chunks
    file_format: str = None,                      # For pre-merged training chunks
    
    dataset_name: str = None,                     # For original train/test CSVs
    dataset_type_to_process: str = "raw",         # "raw" for fitting transformer, "test" for test data
    chunksize_load: int = 50000,                  # Chunksize for reading original CSVs

    # Target column configuration
    target_column: str, # Can be "placeholder" for test data

    # Transformer fitting parameters (used if fitted_transformer is None)
    num_chunks_to_fit_on: int = 5,
    drop_na_during_load: bool = False, # Passed to get_raw_dataset_generator

    # Yielding control
    num_processed_chunks_to_yield: Union[int, str] = "all",

    # Option to pass an already fitted transformer
    fitted_transformer: Optional[RawTransformer] = None
) -> Iterator[Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]]: # Yields X_processed, y_processed, TransactionIDs
    """
    Generator that loads data in chunks (either from pre-merged files or original CSVs),
    fits or uses a RawTransformer, and yields processed (X, y, TransactionIDs) chunks.
    """
    print(f"--- Initializing ETL Generator ---")
    print(f"  Target column: {target_column}")
    print(f"  Fitting on: {num_chunks_to_fit_on} chunks if transformer not provided.")
    print(f"  Yielding: {num_processed_chunks_to_yield} processed chunks.")

    active_raw_transformer: RawTransformer

    if fitted_transformer:
        print("  Using provided pre-fitted RawTransformer.")
        active_raw_transformer = fitted_transformer
    else:
        print("  No pre-fitted transformer. Fitting a new RawTransformer.")
        active_raw_transformer = RawTransformer()
        
        # Determine data source for FITTING the transformer (should always be training-like data)
        fit_data_source_args = {}
        if output_chunk_dir_path and file_format: # Prioritize pre-merged for fitting if available
            print(f"    Fitting transformer using pre-merged chunks from: {output_chunk_dir_path}")
            fit_data_source_args = {
                "output_chunk_dir_path": output_chunk_dir_path,
                "file_format": file_format,
                "target_column": target_column.replace('-', '_'), # Standardize for lookup
                "drop_na": drop_na_during_load
            }
        elif dataset_name: # Fallback to original files for fitting (ensure it's training data)
            print(f"    Fitting transformer using original files for: {dataset_name}, type: 'raw'")
            fit_data_source_args = {
                "dataset_name": dataset_name,
                "target_column": target_column.replace('-', '_'), # Standardize for lookup
                "dataset_type": "raw", # Always fit on 'raw' (training-like) data
                "drop_na": drop_na_during_load,
                "chunksize": chunksize_load
            }
        else:
            raise ValueError("For fitting transformer: provide 'output_chunk_dir_path'/'file_format' OR 'dataset_name'.")

        fit_data_gen = get_raw_dataset_generator(**fit_data_source_args)
        
        initial_raw_chunks_for_fit_X, initial_raw_chunks_for_fit_y = [], []
        print(f"    Loading up to {num_chunks_to_fit_on} raw chunks for fitting transformer...")
        for i, (X_raw_fit_chunk, y_raw_fit_chunk) in enumerate(fit_data_gen):
            if i < num_chunks_to_fit_on:
                if X_raw_fit_chunk is not None and not X_raw_fit_chunk.empty:
                    initial_raw_chunks_for_fit_X.append(X_raw_fit_chunk)
                    if y_raw_fit_chunk is not None: initial_raw_chunks_for_fit_y.append(y_raw_fit_chunk)
                if (i + 1) % 1 == 0 or (i + 1) == num_chunks_to_fit_on:
                     print(f"      Loaded raw chunk {i+1}/{num_chunks_to_fit_on} for fitting (X shape: {X_raw_fit_chunk.shape if X_raw_fit_chunk is not None else 'N/A'}).")
            else: break
        del fit_data_gen
        if not initial_raw_chunks_for_fit_X: raise ValueError("No raw data loaded for fitting transformer.")

        X_for_fit = pd.concat(initial_raw_chunks_for_fit_X, ignore_index=True)
        y_for_fit = pd.concat(initial_raw_chunks_for_fit_y, ignore_index=True) if initial_raw_chunks_for_fit_y and any(c is not None for c in initial_raw_chunks_for_fit_y) else None
        del initial_raw_chunks_for_fit_X, initial_raw_chunks_for_fit_y; gc.collect()
        
        print(f"    Fitting RawTransformer on concatenated data: X={X_for_fit.shape}, y={y_for_fit.shape if y_for_fit is not None else 'N/A'}")
        # Assuming RawTransformer.fit takes X, y DataFrames/Series directly now
        active_raw_transformer.fit(BaseETLData(X=X_for_fit, y=y_for_fit)) # Use BaseETLData if your RawTransformer expects it
        print("    RawTransformer fitting complete.")
        del X_for_fit, y_for_fit; gc.collect()

    # --- Determine data source for PROCESSING and YIELDING chunks ---
    print(f"\n  Streaming data for transformation (processing type: '{dataset_type_to_process}')...")
    process_data_source_args = {}
    if dataset_type_to_process == "test" and dataset_name: # Special handling for original test files
        print(f"    Processing original test files for: {dataset_name}")
        process_data_source_args = {
            "dataset_name": dataset_name,
            "target_column": target_column, # Will be "placeholder" for test
            "dataset_type": "test",
            "drop_na": drop_na_during_load,
            "chunksize": chunksize_load
        }
    elif output_chunk_dir_path and file_format: # Processing pre-merged chunks (likely training)
        print(f"    Processing pre-merged chunks from: {output_chunk_dir_path}")
        process_data_source_args = {
            "output_chunk_dir_path": output_chunk_dir_path,
            "file_format": file_format,
            "target_column": target_column.replace('-', '_'), # Standardize
            "drop_na": drop_na_during_load
            # chunksize for get_raw_dataset_generator for pre-merged files is not used as it reads whole files
        }
    elif dataset_name: # Processing original train/raw files if output_chunk_dir not specified
        print(f"    Processing original '{dataset_type_to_process}' files for: {dataset_name}")
        process_data_source_args = {
            "dataset_name": dataset_name,
            "target_column": target_column.replace('-', '_'), # Standardize
            "dataset_type": dataset_type_to_process, # Should be "raw" if for training
            "drop_na": drop_na_during_load,
            "chunksize": chunksize_load
        }
    else:
        raise ValueError("For processing data: provide 'output_chunk_dir_path'/'file_format' OR 'dataset_name'.")

    main_data_gen = get_raw_dataset_generator(**process_data_source_args)
    
    chunks_yielded_count = 0
    for i, (X_raw_chunk, y_raw_chunk) in enumerate(main_data_gen):
        if X_raw_chunk is None or X_raw_chunk.empty:
            print(f"    Raw chunk {i+1} is None or empty, skipping.")
            continue
        
        # Extract TransactionIDs BEFORE transformation if present (RawTransformer might drop it)
        # get_raw_dataset_generator should already standardize 'TransactionID' if it had a hyphen.
        current_chunk_ids = X_raw_chunk['TransactionID'].copy() if 'TransactionID' in X_raw_chunk.columns else None
        
        print(f"    Transforming raw chunk {i+1} (X shape: {X_raw_chunk.shape}, IDs extracted: {current_chunk_ids is not None})...")
        
        # Ensure y_raw_chunk has the same index as X_raw_chunk for RawTransformer input
        if y_raw_chunk is not None and not X_raw_chunk.index.equals(y_raw_chunk.index):
            y_raw_chunk = y_raw_chunk.set_axis(X_raw_chunk.index) # Align index

        # Transform using the (fitted) active_raw_transformer
        # Pass X, y directly if not using Pydantic BaseETLData for RawTransformer
        transform_input = BaseETLData(X=X_raw_chunk.copy(), y=y_raw_chunk.copy() if y_raw_chunk is not None else None)
        
        processed_chunk_data_item = None
        transform_gen = active_raw_transformer.transform_stream(transform_input)
        for item in transform_gen: # Consume the generator from transform_stream
            if isinstance(item, TransformedData): # Or your actual output type from RawTransformer
                processed_chunk_data_item = item
                break 
        del transform_gen
            
        if processed_chunk_data_item and processed_chunk_data_item.X is not None:
            processed_X_chunk = processed_chunk_data_item.X
            processed_y_chunk = processed_chunk_data_item.y # Will be None if y_raw_chunk was None

            # Ensure y_processed_chunk has same index as X_processed_chunk
            if processed_y_chunk is not None and not processed_X_chunk.index.equals(processed_y_chunk.index):
                 processed_y_chunk = processed_y_chunk.set_axis(processed_X_chunk.index)

            print(f"      Yielding processed chunk {chunks_yielded_count + 1} (X shape: {processed_X_chunk.shape})")
            yield processed_X_chunk, processed_y_chunk, current_chunk_ids
            chunks_yielded_count += 1

            if num_processed_chunks_to_yield != "all" and chunks_yielded_count >= num_processed_chunks_to_yield:
                print(f"  Reached yield limit of {num_processed_chunks_to_yield} processed chunks.")
                break
        else:
            print(f"    Skipping yield for raw chunk {i+1} as transformation did not produce data.")
        
        del X_raw_chunk, y_raw_chunk, transform_input, processed_chunk_data_item
        if current_chunk_ids is not None: del current_chunk_ids
        gc.collect()
            
    del main_data_gen
    print(f"\n  Finished yielding processed chunks from ETL generator. Total yielded: {chunks_yielded_count}")

# Ensure RawTransformer and BaseETLData/TransformedData are defined above this in etl_chain.py
# Also, the get_raw_dataset_generator needs to be the version that handles both
# pre-merged chunks AND original CSVs, and standardizes column names.
# kaggle-workspace
Follow these steps to get your local Kaggle environment up and running:Create the Project Directory: Create the kaggle-workspace directory on your local machine.mkdir kaggle-workspace
cd kaggle-workspace
Create Subdirectories: Create the data, workspace, mlruns, and wandb subdirectories within kaggle-workspace.mkdir data workspace mlruns wandb
Add docker-compose.yml: Create a file named docker-compose.yml inside kaggle-workspace and paste the code from the previous immersive block into it.Add Datasets: Download the datasets for your Kaggle challenges and place them in organized subdirectories within the data/ folder (e.g., data/titanic/, data/house-prices/).Organize Workspace: Create subdirectories within the workspace/ folder for each challenge. This is where you'll save your Jupyter notebooks, scripts, and generated files (outputs, logs, etc.).Start the Environment: Open your terminal or command prompt, navigate to the kaggle-workspace directory, and run the following command:docker compose up -d
This command builds the images (if necessary), creates the containers, and starts the services in detached mode (-d).Access Jupyter: Once the services are running, open your web browser and go to http://localhost:8888. You should see the Jupyter Lab interface. Your data and workspace directories will be available in the file explorer.Access MLflow UI: Open your web browser and go to http://localhost:5000 to access the MLflow tracking server UI.Access W&B UI: Open your web browser and go to http://localhost:8080 to access the Weights & Biases local server UI.Integrate MLflow and W&B in Notebooks:MLflow: In your Jupyter notebooks, you can log experiments, parameters, metrics, and artifacts using the MLflow Python client. Set the tracking URI in your code (though the environment variable in docker-compose.yml should handle this):import mlflow


check if port is allocated locally:
netstat -ano | findstr :8080
sudo lsof -i :8080 -P -n
netstat -tulnp | grep :8080

docker compose up --build -d
docker compose logs jupyter #look for token
http://localhost:8888

docker compose down
docker compose down -v && docker compose up -d


# The tracking URI is already set by the environment variable in docker-compose.yml
# mlflow.set_tracking_uri("http://mlflow:5000")


with mlflow.start_run(run_name="My First Experiment"):
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.85)
    # Log artifacts like models or plots
    # mlflow.log_artifact("model.pkl")
Weights & Biases: Install the wandb library (pip install wandb) in your Jupyter environment (you might need to do this inside the container or build a custom image). Then, initialize a run:import wandb

# Initialize a W&B run
# The WANDB_DIR environment variable points to the shared volume
wandb.init(project="my-kaggle-challenge", name="model-training-run")

# Log parameters, metrics, etc.
wandb.log({"accuracy": 0.92, "loss": 0.15})

# Finish the run
wandb.finish()
You might be prompted to log in the first time you use wandb.init(). Since you're using the local server, follow the instructions to connect to http://localhost:8080.Stop the Environment: When you're done working, stop the containers by running the following command in the kaggle-workspace directory:docker compose down
Making it Dynamic:The dynamism comes from the data/ and workspace/ directory structure and how they are mounted as volumes.Switching Challenges: To work on a different challenge, you simply navigate to the corresponding subdirectory within /home/jovyan/data and /home/jovyan/workspace inside the Jupyter Lab interface. All your work and data for that challenge should be organized there.Persistent Data: Because data, workspace, mlruns, and wandb are mounted as volumes, any files you create or modify within these directories inside the containers will persist on your local machine even after you stop and restart the Docker containers.This setup provides a robust and organized way to manage your Kaggle projects locally with integrated experiment tracking.

import os
from termcolor import colored
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

def create_output_folders(root=""):
    # create these folders if they does not exist. Defaults to current directory
    os.makedirs(root, exist_ok=True)
    results_path = os.path.join(root,"results")
    logs_path = os.path.join(root,"logs")
    data_path = os.path.join(root,"data")
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    if not os.path.isdir(logs_path):
        os.mkdir(logs_path)
    if not os.path.isdir(data_path):
        os.mkdir(data_path)


def train_model(data, model, model_name="", output_dir="", BATCH_SIZE=0, EPOCHS=0):
    print(colored('starting training session...', 'yellow'))
    create_output_folders(root=output_dir)
    checkpointer = ModelCheckpoint(os.path.join(output_dir, "results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join(output_dir, "logs", model_name))
    history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)
    print(colored('training complete', 'green'))
    pass
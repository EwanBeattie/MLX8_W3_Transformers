from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
import numpy as np

class Database():
    def __init__(self):
        self.engine = create_engine('postgresql://donation@localhost/mlx')

    def save_row(self, input_tensor, prediction, true_label):
        with self.engine.connect() as conn:
            conn.execute(text("INSERT INTO mnist_logs (predicted_digit, true_digit, image) VALUES (:pred, :true, :img)"),
                        {"pred": prediction, 
                         "true": true_label, 
                         "img": input_tensor.numpy().astype(np.uint8).tobytes()})
            conn.commit()

    def retrieve_rows(self):
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM mnist_logs"))
            return [Row(row) for row in result]
        
    def delete_all_rows(self):
        with self.engine.connect() as conn:
            conn.execute(text("DELETE FROM mnist_logs"))
            conn.commit()
        

database = Database()

# data = database.retrieve_rows()
database.delete_all_rows()

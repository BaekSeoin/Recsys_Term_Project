# -*- coding: utf-8 -*-
import NeuralMF


def write_output(predict):
    with open('output.txt', 'w') as f:
        for i in predict:
            f.write(i + "\n")

if __name__ == "__main__":
    #predict = NeuralMF.model_train()
    predict = NeuralMF.run_pre_trained_model()
    write_output(predict)



# import os
# import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import load_model
# from pathlib import Path
# import pickle
# from model import MultiHeadAttention
# from spektral.layers import GCNConv, GINConv
# from train import catch
# from evaluation import scores, evaluate
# from tensorflow.keras import utils
# from mamba import MambaBlock,ResidualBlock,Mamba,RMSNorm
# def predict(X_test, K_test, N_test, F_test, A_test, y_test, thred, para, weights, jsonFiles, h5_model, dir):

#     # with open('test_true_label.pkl', 'wb') as f:
#     #     pickle.dump(y_test, f)

#     adam = Adam(learning_rate=para['learning_rate']) # adam optimizer
#     for ii in range(0, len(weights)):
#         # 1.loading weight and structure (model)

#         # json_file = open('BiGRU_base/' + jsonFiles[i], 'r')
#         # model_json = json_file.read()
#         # json_file.close()
#         # load_my_model = model_from_json(model_json)
#         # load_my_model.load_weights('BiGRU_base/' + weights[i])
#         # load_my_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

#         h5_model_path = os.path.join("./", h5_model[ii])
#         tf.keras.backend.clear_session()
#         #load_my_model = load_model(h5_model_path, custom_objects={"MultiHeadAttention" : MultiHeadAttention, "GCNConv": GCNConv, "GINConv": GINConv
#                                                                  #,"MambaBlock":MambaBlock,"ResidualBlock":ResidualBlock,"Mamba":Mamba,"RMSNorm":RMSNorm})
#         #load_my_model = load_model(h5_model_path)
#         load_my_model = tf.keras.models.load_model(h5_model_path)
#         print("Prediction is in progress")

#         # 2.predict
#         X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
#         K_test = tf.convert_to_tensor(K_test, dtype=tf.float32)
#         N_test = tf.convert_to_tensor(N_test, dtype=tf.float32)
#         F_test = tf.convert_to_tensor(F_test, dtype=tf.float32)
#         A_test = tf.convert_to_tensor(A_test, dtype=tf.float32)
#         score = load_my_model.predict([X_test, K_test, N_test, F_test, A_test])

#         "========================================"
#         for i in range(len(score)):
#             for j in range(len(score[i])):
#                 if score[i][j] < thred:
#                     score[i][j] = 0
#                 else:
#                     score[i][j] = 1
#         a, b, c, d, e = evaluate(score, y_test)
#         print(a, b, c, d, e)
#         "========================================"

#         # 3.evaluation
#         if ii == 0:
#             score_label = score
#         else:
#             score_label += score

#     score_label = score_label / len(h5_model)

#     # data saving
#     with open(os.path.join(dir, 'MLBP_prediction_prob.pkl'), 'wb') as f:
#         pickle.dump(score_label, f)

#     # getting prediction label
#     for i in range(len(score_label)):
#         for j in range(len(score_label[i])):
#             if score_label[i][j] < thred: score_label[i][j] = 0
#             else: score_label[i][j] = 1

#     # data saving
#     with open(os.path.join(dir, 'MLBP_prediction_label.pkl'), 'wb') as f:
#         pickle.dump(score_label, f)

#     # evaluation
#     aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(score_label, y_test)

#     print("Prediction is done")
#     print('aiming:', aiming)
#     print('coverage:', coverage)
#     print('accuracy:', accuracy)
#     print('absolute_true:', absolute_true)
#     print('absolute_false:', absolute_false)
#     print('\n')

#     out = dir
#     Path(out).mkdir(exist_ok=True, parents=True)
#     out_path2 = os.path.join(out, 'result_test.txt')
#     with open(out_path2, 'w') as fout:
#         fout.write('aiming:{}\n'.format(aiming))
#         fout.write('coverage:{}\n'.format(coverage))
#         fout.write('accuracy:{}\n'.format(accuracy))
#         fout.write('absolute_true:{}\n'.format(absolute_true))
#         fout.write('absolute_false:{}\n'.format(absolute_false))
#         fout.write('\n')


# def test_my(test, para, model_num, dir):
#     # step1: preprocessing
#     # X_test, K_test, N_test, y_test = test[0], test[1], test[2], test[3]
#     test[5] = utils.to_categorical(test[5])
#     test[0], test[1], test[2], test[3], test[4], temp = catch(test[0], test[1], test[2], test[3], test[4], test[5])
#     temp[temp > 1] = 1
#     test[5] = temp

#     # test[1] = keras.utils.to_categorical(test[1])
#     # test[0], temp = catch(test[0], test[1])
#     # temp[temp > 1] = 1
#     # test[1] = temp

#     # weight and json
#     weights = []
#     jsonFiles = []
#     h5_model = []
#     for i in range(1, model_num+1):
#         weights.append('model{}.hdf5'.format(str(i)))
#         jsonFiles.append('model{}.json'.format(str(i)))
#         h5_model.append('model{}'.format(str(i)))

#     # step2:predict
#     predict(test[0], test[1], test[2], test[3], test[4], test[5], test[6], para, weights, jsonFiles, h5_model, dir)




import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from pathlib import Path
import pickle
from model import MultiHeadAttention,TLULayer,FRNLayer
from spektral.layers import GCNConv, GINConv
from train import catch
from evaluation import scores, evaluate
from tensorflow.keras import utils
from mamba import MambaBlock,ResidualBlock,Mamba,RMSNorm
def predict(X_test, K_test, N_test, F_test, A_test, y_test, thred, para, weights, jsonFiles, h5_model, dir):

    # with open('test_true_label.pkl', 'wb') as f:
    #     pickle.dump(y_test, f)

    for ii in range(0, len(weights)):
        # 1.loading weight and structure (model)

        # json_file = open('BiGRU_base/' + jsonFiles[i], 'r')
        # model_json = json_file.read()
        # json_file.close()
        # load_my_model = model_from_json(model_json)
        # load_my_model.load_weights('BiGRU_base/' + weights[i])
        # load_my_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        h5_model_path = os.path.join("./", h5_model[ii])
        tf.keras.backend.clear_session()
        #load_my_model = load_model(h5_model_path, custom_objects={"MultiHeadAttention" : MultiHeadAttention, "GCNConv": GCNConv, "GINConv": GINConv
                                                                 #,"MambaBlock":MambaBlock,"ResidualBlock":ResidualBlock,"Mamba":Mamba,"RMSNorm":RMSNorm})
        #load_my_model = load_model(h5_model_path)
        print(h5_model_path)
        model.load_weights(h5_model_path)
        print("Prediction is in progress")

        # 2.predict
        score = model.predict([X_test, K_test, N_test, F_test, A_test])

        "========================================"
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] < thred:
                    score[i][j] = 0
                else:
                    score[i][j] = 1
        a, b, c, d, e = evaluate(score, y_test)
        print(a, b, c, d, e)
        "========================================"

        # 3.evaluation
        if ii == 0:
            score_label = score
        else:
            score_label += score

    score_label = score_label / len(h5_model)

    # data saving
    with open(os.path.join(dir, 'MLBP_prediction_prob.pkl'), 'wb') as f:
        pickle.dump(score_label, f)

    # getting prediction label
    for i in range(len(score_label)):
        for j in range(len(score_label[i])):
            if score_label[i][j] < thred: score_label[i][j] = 0
            else: score_label[i][j] = 1

    # data saving
    with open(os.path.join(dir, 'MLBP_prediction_label.pkl'), 'wb') as f:
        pickle.dump(score_label, f)

    # evaluation
    aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(score_label, y_test)

    print("Prediction is done")
    print('aiming:', aiming)
    print('coverage:', coverage)
    print('accuracy:', accuracy)
    print('absolute_true:', absolute_true)
    print('absolute_false:', absolute_false)
    print('\n')

    out = dir
    Path(out).mkdir(exist_ok=True, parents=True)
    out_path2 = os.path.join(out, 'result_test.txt')
    with open(out_path2, 'w') as fout:
        fout.write('aiming:{}\n'.format(aiming))
        fout.write('coverage:{}\n'.format(coverage))
        fout.write('accuracy:{}\n'.format(accuracy))
        fout.write('absolute_true:{}\n'.format(absolute_true))
        fout.write('absolute_false:{}\n'.format(absolute_false))
        fout.write('\n')


def test_my(test, para, model_num, dir):
    # step1: preprocessing
    # X_test, K_test, N_test, y_test = test[0], test[1], test[2], test[3]
    test[5] = utils.to_categorical(test[5])
    test[0], test[1], test[2], test[3], test[4], temp = catch(test[0], test[1], test[2], test[3], test[4], test[5])
    temp[temp > 1] = 1
    test[5] = temp

    # test[1] = keras.utils.to_categorical(test[1])
    # test[0], temp = catch(test[0], test[1])
    # temp[temp > 1] = 1
    # test[1] = temp

    # weight and json
    weights = []
    jsonFiles = []
    h5_model = []
    for i in range(1, model_num+1):
        weights.append('model{}.hdf5'.format(str(i)))
        jsonFiles.append('model{}.json'.format(str(i)))
        h5_model.append('model{}.h5'.format(str(i)))

    # step2:predict
    predict(test[0], test[1], test[2], test[3], test[4], test[5], test[6], para, weights, jsonFiles, h5_model, dir)

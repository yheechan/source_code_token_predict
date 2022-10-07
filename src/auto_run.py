from sklearn.model_selection import train_test_split
import torch

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import timeit

import data
import data_loader as dl
import initializer as init
import trainer
import tester
import predictor
import model_util as mu
import pretrained_model as pm

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

proj_list = [
    'boringssl_total', 'c-ares_total',
    'freetype2_total', 'guetzli_total',
    'harfbuzz_total', 'libpng_total',
    'libssh_total', 'libxml2_total',
    'pcre_total', 'proj4_total',
    're2_total', 'sqlite3_total',
    'total', 'vorbis_total',
    'woff2_total', 'wpantund_total'
]

for i in range(15):

    if (i == 12): continue

    desc = str(i)

    target_project = 0

    prefix_np, postfix_np, label_np = data.getSingleProjectData(proj_list, proj_list[target_project])
    test_prefix, test_postfix, test_label = data.getTestData(proj_list[target_project])


    train_prefix, val_prefix, train_postfix, val_postfix, train_label, val_label = train_test_split(
        prefix_np, postfix_np, label_np, test_size = 0.2, random_state = 43
    )

    # train_prefix, val_prefix, train_postfix, val_postfix, train_label, val_label = train_test_split(
    #     train_prefix, train_postfix, train_label, test_size = 0.2, random_state = 43
    # )


    train_dataloader, val_dataloader, test_dataloader = dl.data_loader(
                                                            train_prefix, train_postfix,
                                                            val_prefix, val_postfix,
                                                            test_prefix, test_postfix,
                                                            train_label, val_label, test_label,
                                                            batch_size=1000
                                                        )


    # PyTorch TensorBoard support
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('../tensorboard/phase2/tests')


    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    # ====================
    # set parameters here
    # ====================

    title = proj_list[target_project] + '_phase2_' + desc
    epochs = 40

    embed_dim = 128
    max_len, source_code_tokens, token_choices = data.getInfo()
    pretrained_token2vec = pm.load_pretrained_model(source_code_tokens, embed_dim)
    pretrained_token2vec = torch.tensor(pretrained_token2vec)


    input_size = max_len
    hidden_size = 200
    num_classes = max(token_choices) + 1
    rnn_layers = 2

    num_filters = [100, 200, 100]
    kernel_sizes = [15, 21, 114]

    dropout = 0.3

    learning_rate = 0.001
    weight_decay = 0

    model_name = "RNN"
    optim_name = "Adam"
    loss_fn_name = "CEL"

    pretrained_model = pretrained_token2vec
    freeze_embedding = False,


    trainer.set_seed(42)

    model, optimizer, loss_fn = init.initialize_model(
        vocab_size=input_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_classes=num_classes,
        rnn_layers=rnn_layers,
        num_filters=num_filters,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        model_name=model_name,
        optim_name=optim_name,
        loss_fn_name=loss_fn_name,
        pretrained_model=pretrained_model,
        freeze_embedding=freeze_embedding,
        device=device,
    )

    print(model)

    start_time = timeit.default_timer()

    trainer.train(
        epochs=epochs,
        title=title,
        writer=writer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn
    )

    end_time = (timeit.default_timer() - start_time) / 60.0

    mu.saveModel(title, model)

    model = mu.getModel(title)
    print(model)


    loss, acc = tester.test(test_dataloader=test_dataloader,
                            device=device,
                            model=model,
                            title=title)
    

    with open('../result/phase2', 'a') as f:
        text = title + '\t |\tloss: ' + str(loss) + '\t |\tacc: ' + str(acc) + '\t |\t time: ' + str(round(end_time, 3)) + ' min\n'
        f.write(text)
    

    mu.graphModel(train_dataloader, model, writer)


def save_pair_bert(output_predict_file, predict_datasets, labels_idx, predictions, logs):
    num_labels = len(logs)
    with open(output_predict_file, "w") as writer:
        if num_labels == 2:
            logs0, logs1 = logs[0], logs[1]
            writer.write("title\tprediction-idx\treal-label\tlog0\tlog1\n")
            for row, pred, label_idx, log0, log1 in zip(predict_datasets, predictions, labels_idx, logs0, logs1):
                guid = row['title'].split(':')[-1]
                real_label = label_idx
                writer.write(f"{guid}\t{pred}\t{real_label}\t{log0}\t{log1}\n")

        elif num_labels == 4:
            logs0, logs1, logs2, logs3 = logs[0], logs[1], logs[2], logs[3]
            writer.write("title\tprediction-idx\treal-label\tlog0\tlog1\tlog2\tlog3\n")
            for row, pred, label_idx, log0, log1, log2, log3 in zip(predict_datasets, predictions, labels_idx, logs0, logs1, logs2, logs3):
                guid = row['title'].split(':')[-1]
                real_label = label_idx
                writer.write(f"{guid}\t{pred}\t{real_label}\t{log0}\t{log1}\t{log2}\t{log3}\n")

        else: #num_labels = 5
            logs0, logs1, logs2, logs3, logs4 = logs[0], logs[1], logs[2], logs[3], logs[4]
            writer.write("title\tprediction-idx\treal-label\tlog0\tlog1\tlog2\tlog3\tlog4\n")
            for row, pred, label_idx, log0, log1, log2, log3, log4 in zip(predict_datasets, predictions, labels_idx, logs0, logs1, logs2, logs3, logs4):
                # item = label_list[item]
                guid = row['title'].split(':')[-1]
                real_label = label_idx
                # text = row['text']
                # writer.write(f"{guid}\t{text}\t{pred}\t{real_label}\t{log0}\t{log1}\t{log2}\t{log3}\t{log4}\n")
                writer.write(f"{guid}\t{pred}\t{real_label}\t{log0}\t{log1}\t{log2}\t{log3}\t{log4}\n")
    return 
        
    


def save_pair_t5(output_predict_file, predict_datasets, labels_idx, predictions):
    with open(output_predict_file, "w") as writer:
        writer.write("guid\tprediction-idx\treal-label\n")
        for row, pred, label_idx in zip(predict_datasets, predictions, labels_idx):
            # item = label_list[item]
            guid = row['title'].split(':')[-1]
            real_label = label_idx
            # text = row['text']
            writer.write(f"{guid}\t{pred}\t{real_label}\n")

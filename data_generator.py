import os
import tensorflow as tf
import csv
import tokenization
from run_classifier import InputExample

class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""
  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = tokenization.convert_to_unicode(line[0])
      if language != tokenization.convert_to_unicode(self.language):
        continue
      text_a = tokenization.convert_to_unicode(line[6])
      text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[8])
      text_b = tokenization.convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

# class AmazonProcessor():
#   def __init__(self, num_sample_per_class, batch_size):
#     self.batch_size = batch_size
#     self.num_sample_per_class = num_sample_per_class
#     self.num_classes = 2
  
#    def _divide_tasks(data_name, task_list, name='train'):
#     task_class = ['t2', 't5', 't4']
#     for task in task_class:
#         file_name = data_name+'.'+task
#         # print(file_name)
#         task_sent, task_labels = self._load_single_data(file_name+'.'+name)
#         task_list.append({'sentence': task_sent, 'label': task_labels})

#   def _load_single_data(data_name):
#     with open(data_name, 'r') as f:
#       data_lines = f.readlines()
    
#     new_lines = [line.strip().split('\t') for line in data_lines]
#     labels = [line[1] if int(line[1]) == 1 else '0' for line in new_lines]
#     sentences = [line[0] for line in data_lines]

#     return sentences, labels
  
#   def get_train_examples(self, data_dir):
    
        
    

class AmazonProcessor(DataProcessor):
  """Processor for the Amazon data set ."""
  def calculate_task_num(self,data_dir):
    with open(data_dir+'/workspace.filtered.list', 'r') as train_f:
      train_list = train_f.readlines()
    self.train_task_num = len(train_list)*3

    with open(data_dir+'/workspace.target.list', 'r') as test_f:
      test_list = test_f.readlines()
    self.test_task_num = len(train_list)*3
    
    return self.train_task_num, self.test_task_num

  def _read_file(self,dataname):
    
    with tf.gfile.Open(dataname, "r") as f:
      reader = csv.reader(f, delimiter="\t")
      lines = []
      for line in reader:
        lines.append(line)
    return lines

  def _divide_tasks(self,data_name, type):
    task_class = ['t2', 't5', 't4']
    tasks = []
    total_len = 0
    for task in task_class:
      file_name = data_name+'.'+task+'.'+type
      print(file_name)
      task_data = self._read_file(file_name)
      tasks.append(task_data)
      total_len = total_len+len(task_data)
    
    return tasks, total_len

  def _get_examples(self, data_dir, filter_name, type):
  
    tasks = []
    total_len = 0
    with open(os.path.join(data_dir, filter_name), 'r') as f:
      task_list = f.readlines()
      task_list = [name.strip() for name in task_list]
    for task_name in task_list:
      diverse_task, task_len = self._divide_tasks(os.path.join(data_dir, task_name), type)
      tasks.extend(diverse_task)
      total_len += task_len
    return tasks, total_len

  def load_all_data(self, data_dir):
    self.train_tasks,self.train_number = self._get_examples(data_dir, 'workspace.filtered.list', 'train')
    self.dev_tasks,self.dev_number = self._get_examples(data_dir, 'workspace.filtered.list', 'dev')
    self.test_tasks,self.test_number = self._get_examples(data_dir, 'workspace.filtered.list', 'test')
    self.fsl_train,self.fsl_train_number = self._get_examples(data_dir, 'workspace.target.list', 'train')
    self.fsl_dev,self.fsl_dev_number = self._get_examples(data_dir, 'workspace.target.list', 'dev')
    self.fsl_test,self.fsl_test_number = self._get_examples(data_dir, 'workspace.target.list', 'test')
      

  def get_train_examples(self, data_dir, task_id=0):
    return self._create_examples(self.train_tasks[task_id],"train",task_id) 

  def get_dev_examples(self, data_dir,task_id=0):
    """See base class."""
    return self._create_examples(self.dev_tasks[task_id],"dev",task_id) 

  def get_test_examples(self, data_dir,task_id=0):
    """See base class."""
    return self._create_examples(self.test_tasks[task_id],"test",task_id)

  def get_fsl_train_examples(self, data_dir, task_id=0):
    
    return self._create_examples(self.fsl_train[task_id],"train",task_id) 

  def get_fsl_dev_examples(self, data_dir,task_id=0):
    """See base class."""
    return self._create_examples(self.fsl_dev[task_id],"dev",task_id) 

  def get_fsl_test_examples(self, data_dir,task_id=0):
    """See base class."""
    return self._create_examples(self.fsl_dev[task_id],"test",task_id)


  def get_labels(self):
        
    """See base class."""
    return ["-1", "1"]

  def _create_examples(self, lines, set_type, set_id):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):    
      guid = "%s-%s-%s" % (set_type,set_id, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[0])
        label = "-1"
      else:
        text_a = tokenization.convert_to_unicode(line[0])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

from __future__ import division
from corpus import Document, NamesCorpus, ReviewCorpus
from maxent import MaxEnt
import matplotlib.pyplot as plt

class BagOfWords(Document):
    def features(self):
        """Trivially tokenized words."""
        return "bagofwords"

class Name(Document):
    def features(self):
        name = self.data
        return ['First=%s' % name[0], 'Last=%s' % name[-1]]

class Bigram(Document):
    def features(self):
        #different features generating mode
        return "bigram"

classifier = MaxEnt()
instances = ReviewCorpus('yelp_reviews.json',document_class=BagOfWords)

# ##experiment 1
# print('experiment 1')
# y1 = []
# x = []
# lengths = [1000,10000,50000,100000,len(instances.documents)]
# for length in lengths:
#     score = classifier.train(instances, maxlength=length, batch_size=30, l2_value=0.1, dev_instances=None)
#     print("score:",score)
#     y1.append(score)
#     x.append(str(length))
#
# plt.plot(x,y1)
# for xy in zip(x, y1):
#     plt.annotate("(%s,%s)" % xy, xy=xy, xytext=(-20, 10), textcoords='offset points')
#
# plt.savefig("experiment1.jpg")
# plt.show()
#
# ##experiment 2
# print('experiment 2')
# batch_sizes = [1,10,50,100,1000]
# y2 = []
# x = []
# for batch_size in batch_sizes:
#     score = classifier.train(instances, maxlength=50000, batch_size=batch_size, l2_value=0.1, dev_instances=None)
#     print("score:",score)
#     y2.append(score)
#     x.append(str(batch_size))
#
# plt.plot(x,y2)
# for xy in zip(x, y2):
#     plt.annotate("(%s,%s)" % xy, xy=xy, xytext=(-20, 10), textcoords='offset points')
#
# plt.savefig("experiment2.jpg")
# plt.show()
#
#
# ##experiment 3
# print('experiment 3')
# y3 = []
# x = []
# l2 = [0.1,0.15,0.2,0.5,1,5,10]
# for l in l2:
#     score = classifier.train(instances, maxlength=50000, batch_size=30, l2_value=l, dev_instances=None)
#     print("score:", score)
#     y3.append(score)
#     x.append(str(l))
#
# plt.plot(x,y3)
# for xy in zip(x, y3):
#     plt.annotate("(%s,%s)" % xy, xy=xy, xytext=(-20, 10), textcoords='offset points')
#
# plt.savefig("experiment3.jpg")
# plt.show()
#
# #Experiment 4 - Bigram
# classifier = MaxEnt()
# instances = ReviewCorpus('yelp_reviews.json',document_class=Bigram)
# print('experiment 4')
# y4 = []
# x = []
# lengths = [1000,10000,50000,100000,len(instances.documents)]
# for length in lengths:
#     score = classifier.train(instances, maxlength=length, batch_size=30, l2_value=0.1, dev_instances=None)
#     print("score:",score)
#     y4.append(score)
#     x.append(str(length))
#
# plt.plot(x,y4)
# plt.plot(x,y1)
# for xy in zip(x,y4):
#     plt.annotate("(%s,%s)" % xy, xy=xy, xytext=(-20, 10), textcoords='offset points')
# for xy in zip(x,y1):
#     plt.annotate("(%s,%s)" % xy, xy=xy, xytext=(20, -10), textcoords='offset points')
# plt.savefig("experiment4.jpg")
# plt.show()

#best : length = full, batch = 50, l2 = 0.2

score = classifier.train(instances, maxlength=len(instances.documents), batch_size=50, l2_value=0.2, dev_instances=None)
print("score:",score)

#predict input comment
while True:
    inp = input('type a string to test its tag,enter q to exit:\n')
    if inp == 'q' or inp == 'Q':
        break
    result = classifier.classify(inp)
    print(result)
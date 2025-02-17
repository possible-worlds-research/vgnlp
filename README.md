# VGNLP

VGNLP is a package to download and process the [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) dataset. The Visual Genome (VG) is an annotated image dataset containing over 100,000 images and millions of region descriptions, visual question-answer pairs, as well as attribute and relations. It is an excellent resource to train models that take reference phenomena seriously.   

The scripts in this repository are geared towards training tiny language models to acquire basic reference capabilities.

The Visual Genome by Ranjay Krishna is licensed under a Creative Commons Attribution 4.0 International License.

## Extracting data

The *extract.zip* script will download parts of the VG dataset, namely the question-answer pairs and region descriptions. It then processes the files and creates two folders: *obs* and *skt*. 

The *obs* (for 'observation') folder contains region description data in the following format: 

```
<a type=OBS idx=1>
<e>the clock is green in colour</e>
<e>shade is along the street </e>
<e>man is wearing sneakers</e>
<e>cars headlights are off</e>
...
</a>

<a type=OBS idx=2>
<e>walk sign is lit up</e>
<e>man wearing silver backpack</e>
<e>red car parked on road</e>
<e>3 green trees line street</e>
<e>crosswalk</e>
<e>A bricked sidewalk. </e>
...
</a>
```

'a type' stands for 'activity type'. It tells us that in this dataset, the model is simply observing the data. Its observations consist in sets of entities, marked by the 'e' tags. The descriptions inside the entities correspond to the region descriptions in the VG.

The *skt* (for 'skill training') folder contains question-answer pairs in the following format:

```
<a type=SKT idx=1>
<u speaker=HUM>Where is the white work truck?</u>
<u speaker=BOT>Parked on the street.</u>
</a>
<a type=SKT idx=1>
<u speaker=HUM>Where is the black sign?</u>
<u speaker=BOT>On the front of the building.</u>
</a>
...
<a type=SKT idx=2>
<u speaker=HUM>Where are cars?</u>
<u speaker=BOT>Parked on the street.</u>
</a>
...

```

Again, 'a type' stands for 'activity type'. It tells us that in this dataset, the model is training a skill, namely question answering. For each observation in the *obs* folder, we obtain a set of question-answer pairs relating to that observation.


## Making training data for a language model

Running the script *mktrain.py* will produce training data that can be fed to a language model. The script takes two arguments. The first argument corresponds to the size of the 'memory' slot given to the model: how many entities from a situation does the model actually remember? The second argument is the number of training examples that should be generated.

So for instance:

```
python3 mktrain.py 3 5
```

will produce 5 training instances consisting of 3 remembered entities from a situation and the associated question-answer pair. It is possible that the remembered entities (which are randomly sampled) do not correspond to the entity that the question refers to:

```
<a type=OBS>
<e>a wall panel control</e>
<e>tv hanging on white wall</e>
<e>thermostat on the wall</e>
</a>
<a type=TLK>
<u speaker=HUM>Who is wearing a sweater?</u>
<u speaker=BOT>The guy on the left.</u>
</a>
<a type=OBS>
<e>van has rear doors open</e>
<e>the bus is red</e>
<e>side door of van is open</e>
</a>
<a type=TLK>
<u speaker=HUM>How is the bus that has stopped at a bus stop?</u>
<u speaker=BOT>The bus is color red.</u>
</a>
<a type=OBS>
<e>black table and chairs</e>
<e>white sign on green wall</e>
<e>a part of a black trashbag. </e>
</a>
<a type=TLK>
<u speaker=HUM>What cup of french fries on the table?</u>
<u speaker=BOT>Fries in red and white cup.</u>
</a>
<a type=OBS>
<e>black arrow pointing right</e>
<e>stormy skies over wind mills</e>
<e>white lines on the edge of the road</e>
</a>
<a type=TLK>
<u speaker=HUM>Where is the yellow sign?</u>
<u speaker=BOT>Under the stop sign.</u>
</a>
<a type=OBS>
<e>a bright light in the background</e>
<e>a back view of a street sign</e>
<e>photo was taken at night</e>
</a>
<a type=TLK>
<u speaker=HUM>Where is the photographer?</u>
<u speaker=BOT>In front of the hydrant.</u>
</a>
```

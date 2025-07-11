# Vector Database
We assume that the word 'mother' is a abtract identity for te underlying sense data:
- sound of mother
- image of mother
- touch of mother
- smell of mother
- feelings
- etc.

Initially, we get for example, 4 of such vector data simultaneously. Then we create an high level abtraction for this idea 'mother'.

Now save teh vector data for each sense data in teh vector database.
NExt step is to save the abstraction identity 'mother'. We create unique vector and save it to teh vector database. Now we need to create connection. This will be usually in teh form of metadata with children ids of a parent 'mother', their connection weights. Also for each child, their children ids and weights and parent ids and weights.

Now when we get a touch vector and search for it in teh vector database, we get the entry and id. the use this id to get the metadata stored external to vector database. Now we can get the parent and children connections. for example, mother and then get all teh children of teh mother. now we can recreat, how the 'mother' identity.

Same now we can do with teh identity mother. We can search with corresponding vector and get the id and then from external metaata get all connection information.

We can also go one step further, where we can detatch identity from language and instead of mother, use some id, then map word mother to this id.
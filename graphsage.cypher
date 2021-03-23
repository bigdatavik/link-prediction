CALL gds.graph.create(
  'authors',
  {
    Author: {
      label: 'Author',
      properties: ['coefficientTest', 'louvainTest', 'partitionTest', 'trianglesTest']
    }
  }, {
    CO_AUTHOR: {
      type: 'CO_AUTHOR',
      orientation: 'UNDIRECTED',
      properties: ['collaborations']
    }
})


CALL gds.beta.model.drop('authors')


# You can run train multiple times

CALL gds.beta.graphSage.train(
  'authors',
  {
    modelName: 'authors',
    featureProperties: ['coefficientTest', 'louvainTest', 'partitionTest', 'trianglesTest'],
    aggregator: 'mean',
    activationFunction: 'sigmoid',
    sampleSizes: [25, 10],
    degreeAsProperty: true,
    embeddingDimension: 3,
    relationshipWeightProperty: 'collaborations'
  }
)




CALL gds.beta.graphSage.stream(
  'authors',
  {
    modelName: 'authors'
  }
)


CALL gds.beta.graphSage.write(
  'authors',
  {
    writeProperty:'graphsage_embedding',
    modelName: 'authors'
  });




  Calculate Similarity embedding


MATCH (c:Author)
WITH {item:id(c), weights: c.graphsage_embedding} AS userData
WITH collect(userData) AS data
CALL gds.alpha.similarity.cosine.stream({
 data: data,
 skipValue: null
})
YIELD item1, item2, count1, count2, similarity
RETURN gds.util.asNode(item1).name AS from, gds.util.asNode(item2).name AS to, similarity
ORDER BY similarity DESC



MATCH (i:Author)
 WITH {item:id(i), weights: i.graphsage_embedding} AS itemData
 WITH collect(itemData) AS data
 CALL gds.alpha.similarity.cosine.write({
  data: data,
  skipValue: null,
  topK: 5,
  similarityCutoff:.1,
  writeRelationshipType:'SIMILAR_GRAPHSAGE_EMBEDDING'
 })
YIELD min, max, mean, stdDev, p25, p50, p75, p90, p95, p99, p999, p100
RETURN min, max, mean, stdDev





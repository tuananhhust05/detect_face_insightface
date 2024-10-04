from elasticsearch import Elasticsearch
import numpy as np

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Define the index name
index_name = "images"

# # Function to insert a document into the index
# def insert_document(index, doc_id, title, vector):
#     doc = {
#         "title": title,
#         "content_vector": vector
#     }
#     es.index(index=index, id=doc_id, body=doc)
#     print(f"Document {doc_id} inserted.")

# # Example document data
# doc_id = "1"
# title = "Sample Document"
# # Example vector - make sure it matches the dimensions defined in your mapping
# vector = [1.34199965, 0.372708827, -0.0588477775, -0.89345789, 0.610700309, 0.525113106, 0.884946108, 0.886321843, -0.298181176, 0.983158767, 0.568788, 1.45619476, -1.45707548, -0.632924318, 0.174408451, 0.377373338, -0.527926862, 2.06872439, 0.426714748, 0.659407675, -0.195948452, 1.55839717, 0.992295086, 0.789074361, 0.0375448503, -0.911133111, -0.438989788, -1.60747528, -0.459385723, 0.0708008558, 0.673009753, 0.169843882, 0.239460155, -0.213576809, -0.530890048, -0.0229561441, -0.156489328, 0.38511011, -0.349697649, 0.265504271, 0.086432904, -1.51654077, -1.42805433, -0.662253737, -0.0701519921, -0.298402399, 0.374622762, -0.554539621, -0.213901103, 0.862473428, 0.218232065, 0.600528717, 0.479800701, 0.432103604, 0.623395085, 0.157770529, 0.10215728, -0.562871456, -1.22289538, 0.782706678, -0.63602829, -0.532271, -0.13722752, 1.58510685, -1.54905832, -0.403393626, 0.578781307, 0.936083376, -0.728851259, 1.00236332, 1.05352294, 0.733387887, -1.03767467, -0.217726395, -0.186108455, 1.15222704, 0.506929815, 1.79479396, -0.770385563, -0.692156255, -1.08299041, -0.509101331, -0.335369229, 0.878121078, -0.535518646, -0.665573239, 0.128091246, -0.0155227855, -0.97137922, 0.716488838, -1.31388223, 1.26459754, 0.361606807, -0.837635338, 0.876948953, -0.323539644, -1.39559937, -0.368136555, -0.949307382, -0.529679537, -0.690772772, -0.0132163716, -0.426724, -0.223463967, -0.0399375036, 0.549505353, -0.040888764, -0.350842774, -0.790020406, 0.3922095, -1.83703613, -0.690307438, 0.312803209, 0.246124342, 0.242630407, -0.201492816, 0.475440651, -1.40973759, 1.37914085, -0.672863066, 0.945522904, 0.00513364514, 0.934858739, 0.955113232, -0.361270905, 1.01383591, -0.303876519, 0.333351582, -0.979213774, 0.639567494, 1.72953284, 0.634144723, -1.54139578, -1.82291317, -1.65551054, -0.267294198, 1.07457542, -0.241758615, 0.433990657, 0.914306164, -0.231048211, -0.534582675, -0.379936427, 0.147842377, 0.726311862, -0.69515419, 0.479832798, 0.0188345984, 0.511094332, -1.10739815, 0.625032127, 0.172554046, -0.556350708, -0.424586445, -0.50561291, -0.640953898, -1.84475744, 0.471115947, 1.44885325, 0.204608947, -0.425503582, 0.0666616037, -1.22379518, 0.502447128, -0.548003256, 1.17921436, -0.440020978, 0.790409625, 0.617017865, 0.8640365, 0.917067587, -0.643406749, 0.489125609, 0.995809615, -1.63102269, 1.24077809, -0.138720974, -0.576688886, 0.536709547, -0.333175451, 0.616805077, 0.130802795, -1.7342186, -1.2659018, 0.39372775, 0.786794484, 0.10195794, -0.389845729, -0.897316515, 0.0246155895, -0.261318535, -0.95887661, -1.75169468, -0.681014061, -0.0376218855, -1.07830381, -0.363637537, 0.484997123, -0.246911317, 1.29754817, 1.73859775, 0.641887307, -0.361593485, 0.593843, 1.28350842, 0.643953323, 0.660104871, -0.5526824, 0.0894214958, -0.880959749, 1.67844462, -0.193196073, 0.282271653, 0.337803215, 0.525471032, -0.370148271, 0.272827387, -0.159503028, -0.572440922, 1.14614511, 0.803336918, 1.12403131, 1.30957651, -0.208623394, -1.2017206, 0.0828624815, -0.466151327, 0.445498139, -1.63244212, -0.42985031, -0.989398479, 0.0194141977, 2.45901394, -1.05334, 0.311706692, -0.172417313, -0.65032649, -1.79438674, -0.00235958025, 0.365287483, -1.81929731, 0.946346, 0.236859262, -0.117136821, 0.768669, 0.144928709, 0.88913554, 1.51235473, 1.59065545, -0.654203355, 1.31786454, -1.0378015, -0.494993508, -0.564244032, -0.467701644, -0.169775933, -0.112570673, -1.32286513, 1.07153916, 0.229850739, 0.263444901, -0.404436231, -1.17065036, 0.302726567, 0.349274218, 0.698547, 0.130845413, 1.4610877, -0.113468423, -0.232927397, -0.0349766277, 0.0508535653, 1.11020577, 0.732469916, 2.28506875, 0.597453594, 0.353883773, -1.62159538, -0.389775217, 0.138912648, -0.0360332765, -0.147453383, 1.10368669, 0.431333899, -0.851439178, -0.19497785, 0.695749879, 0.528045177, 0.864381135, -0.425746948, 1.60932374, 0.275650889, 1.00887978, 0.487743914, 0.809908092, -0.727688789, -0.0273163673, -0.0658401772, -0.842440307, 0.638773501, -0.114897504, 0.122287124, 0.155423462, 0.732042968, -0.594221056, -0.198442429, 1.0497998, 0.0189418849, -0.508079529, 0.359324872, 0.668668389, -1.01102805, 1.37903547, 0.403264284, -0.39098689, 0.0941076726, -1.2102865, -0.267732501, -0.296538323, 0.0406278558, -0.730673909, 0.583327234, 0.167077899, 0.808817506, 0.435425282, 1.07815671, 0.135361418, 0.177154526, -0.496019602, 0.681353629, 0.649052858, 0.482336134, -0.426257491, -0.0054905829, -0.465191454, 0.489724, -0.714395642, -0.275972635, -1.85804701, 0.0484315753, -1.62572193, 0.29461813, -0.638209045, -0.165326759, 1.47742808, 1.57111943, 0.147915259, 0.564504, -1.81146419, 0.0925648287, -0.455565453, 0.0923627689, 0.159290045, 1.75775599, 0.42157039, -0.648352683, 0.0367427692, 1.1556586, -0.174443394, 2.36432099, -0.752428353, 0.541762948, -0.00504970364, 0.117804281, 0.291123629, 0.641342044, 1.35626256, 0.601130366, 0.306865811, -0.520988405, -1.21349132, -1.04499817, 0.598221421, -0.502337158, 0.221341789, 2.05970049, -0.682721674, -0.528032362, -0.368701696, -0.800362527, -0.314416915, 0.0907977745, -0.927212059, -0.406604707, 0.368108451, 0.857820332, 0.369654298, 0.372992098, 0.0551452786, -0.0952133462, -0.672504425, -0.0891307294, 0.00675488124, -0.373058468, 0.0996878296, 1.36525273, 0.413384, 1.25884473, -0.551861167, -0.131319717, -0.58409518, 0.411769092, -0.0761755928, -0.555645764, -1.23945296, 0.160949051, 0.846830249, -1.45763826, 0.466974944, 0.812426925, 1.59374309, -0.230901405, -0.659347355, 0.889032245, 0.593995392, -0.742816508, 0.360309422, 0.783577621, -0.424346, 0.682483137, 0.0270524062, -0.180944711, 0.0895314142, -0.613333702, 0.734806716, -1.65366125, 0.175530404, -0.535547316, -2.14986515, 0.226798847, -0.974189878, 0.31091103, -0.455956399, 1.77505171, -0.0244079735, -0.770553052, -0.206342116, 0.473150164, -0.230032787, 1.07819176, 1.00238919, 0.273487121, -0.596565366, 0.53016609, 0.949056387, 0.135761559, 0.809727311, -0.238726482, -1.15641642, 0.426547825, 0.292208523, 0.201454729, -0.845837712, 0.127626687, 0.143997371, 0.270166069, 0.419152856, -1.52295852, 0.469202936, -0.219807163, 0.195235312, 0.301255, -0.382151, -1.30536175, 0.909940481, -1.6048981, 0.414733261, -0.975259304, -0.698304117, 0.142740235, 1.37696469, -0.335918427, 0.185094, 1.10899675, -0.212776721, 0.107344784, 1.12403619, -0.284751356, -0.766344309, 0.132704929, 0.530429065, 0.215175703, 0.357429296, 0.914016247, 0.44127211, -0.888761163, -0.113750286, 0.947725058, 0.262667567, 0.107200988, 0.169795349, -0.19398059, 0.374192625, -0.013261837, -0.967998803, -0.164759383, 0.122833215, 0.743142962, 0.738631248, -1.18957829, -1.46355176, -1.16519296, 0.60634917, -0.021290902, -0.102766842, -0.292631298, -1.04470897, -0.447630256, 0.387875646, -0.783450902, -1.40529382, -0.125380471]

# # Insert the document
# insert_document(index_name, doc_id, title, vector)

# # Define a query vector (same dimensions)
query_vector = np.random.rand(512).tolist()

# Function to perform a cosine similarity search
def search_with_cosine_similarity(index, query_vec):
    search_query = {
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                    "params": {
                        "query_vector": query_vec.tolist()
                    }
                }
            }
        }
    }

    response = es.search(index=index, body=search_query)
    return response

# Perform the search
response = search_with_cosine_similarity(index_name, query_vector)

# Display the results
print("Search Results:")
for hit in response['hits']['hits']:
    print(f"ID: {hit['_id']}, Score: {hit['_score']}, Title: {hit['_source']['title']}")

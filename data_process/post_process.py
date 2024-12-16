import json
import sys
import pandas as pd
from util import pull_data

def process(pred_df: pd.DataFrame, class_node_map: dict):
    print("Post Processing...")
    
    class_processed_data = pull_data(
        '''
        select `index`, projectName, moduleName, packageName, fullClassName from class_processed_data
        ''',
        'inference'
    )

    result = {}

    for index, row in pred_df.iterrows():
        targetID = int(row["group_id"])
        projectName = class_processed_data.where(f'index == {targetID}').select('projectName').toPandas().iloc[0, 0]
        if projectName not in result:
            result[projectName] = {}

        targetFullClassName = class_processed_data.where(f'index == {targetID}').select('fullClassName').toPandas().iloc[0, 0]
        if targetFullClassName not in result[projectName]:
            result[projectName][targetFullClassName] = []
        
        neiborID = class_node_map[str(int(row["node_id"]))]
        if neiborID == targetID:
            continue
        
        neiborFullClassName = class_processed_data.where(f'index == {neiborID}').select('fullClassName').toPandas().iloc[0, 0]
        if neiborFullClassName not in result[projectName][targetFullClassName]:
            result[projectName][targetFullClassName].append(neiborFullClassName)
        
        print(projectName, '|', targetFullClassName, '|', neiborFullClassName)

    print("Done!")

    return result

if __name__ == "__main__":
    if len(sys.argv) > 2:
        model_pred_path = sys.argv[1]
        class_node_map_path = sys.argv[2]
    else:
        raise ValueError("Please provide the path to the model prediction results.")
    
    df = pd.read_csv(model_pred_path)
    
    with open(class_node_map_path, 'r') as f:
        class_node_map = json.load(f)
    
    result = process(df, class_node_map)

    with open('final_result.json', 'w') as f:
        json.dump(result, f)
import pandas as pd


def kaggle_csv_to_tps(input_file, output_file, name):
    df = pd.read_csv(input_file, engine='python')

    tps_str = 'NAME : %s\nTYPE : TSP\nDIMENSION : %d\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n' % (name, len(df.CityId))

    for c_id, x, y in zip(df.CityId, df.X, df.Y):

        tps_str += "%d %f %f \n" % (c_id, x, y)

    tps_str += "EOF"

    with open(output_file, "w") as f:
        f.write(tps_str)


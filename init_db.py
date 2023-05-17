import pandas as pd
import numpy as np 
import psycopg2
import psycopg2.extras as extras

conn = psycopg2.connect( dbname='epicdb',host='dpg-chihf7pmbg557hesh2c0-a.singapore-postgres.render.com',user='epicdb_user',password='WFNALRLUpMeL27VekaA9igBgrUETKoSV',port='5432')

curr= conn.cursor()

def clean_data(df):
  
  df.columns= df.columns.str.strip().str.lower()
  
  #remove a. 
  alphabet = list(string.ascii_lowercase)
  columns = list(df)
  for col in columns:
    for i, j in enumerate(df[col]):
      if type(j)==type('aaaa'):
        if len(j)<3:
          continue
        if j[0] in alphabet and j[1:3]==". ":
          df[col][i]=j[3:]
  #adding no to null value      
  for i, j in enumerate(df['wandering_headbanging_observation']):
      if df['wandering_headbanging_observation'][i]!='No' and df['wandering_headbanging_observation'][i]!='Yes':
         df['wandering_headbanging_observation'][i]='No'
            
  #removing uncertain value
  f = df.loc[df['final_diagnosis'] == 'Uncertain'].index
  df.drop(f, inplace = True)
  
  df['patientage'] = df['patientage'].astype(float)
  #convert age from months to years
  for i,age in enumerate(df['patientage']):
    if type(age)==type('aa'):
        s=""
        s1=""
        j=0
        for alpha in age:
          if alpha >='0' and alpha<='9':
            if(alpha=='0' and j!=0):
                s+=alpha
          elif alpha >='a' and alpha<='z':
            s1+=alpha
          j+=1
        s1=s1.lower()
          
        if s1=='month' or s1=='months':
          df['patientage'][i]=float(s)/12
        else:
           df['patientage'][i]=float(s)
  
  

  return df
def db_init():
    df = pd.read_excel('Copy_epileptic_seizures_Responses_New.xlsx')
    df = clean_data(df)
    cols=list(df)
    # print(cols)
    sql=''' CREATE TABLE epilepsydata (
        caseno TEXT,
        patientage TEXT,
        gender TEXT,
        eventDuration TEXT,
        stereotypic TEXT,
        events_occur_time TEXT,
        wandering_headbanging_observation TEXT,
        eyes_closed_during_event TEXT,
        weeping_before_during_after_episode TEXT,
        patient_fall_suddenly_without_limb_movements TEXT,
        Was_patient_hyperventilating TEXT,
        loss_consciousness_after_urination_defecation TEXT,
        side_to_side_head_nodding_pelvic_thrusting_Opisthotonic_posturing TEXT,
        observation_limbjerking TEXT,
        observe_postevent_stridulous_laboured_breathing TEXT,
        upper_limb_jerks_observation TEXT,
        brieflossoftouch TEXT,
        fixed_Aura_Premonition TEXT,
        staring_blankly_chewing_smacking_lips TEXT,
        posturing_limbeyehead_deviation TEXT,
        violent_thrashing_movements TEXT,
        recovery TEXT,
        bittentongue TEXT,
        urine_without_knowledge TEXT,
        dislocated_shoulder TEXT,
        other_injuries TEXT,
        final_diagnosis TEXT
    );'''

    curr.execute(sql)
    tuples = [tuple(x) for x in df.to_numpy()]
    cols = ','.join(list(df.columns))
    
    # SQL query to execute
    table='epilepsydata'
    query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
    try:
        extras.execute_values(curr, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
db_init()
conn.commit()
curr.close()
conn.close()

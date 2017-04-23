# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np

energy = pd.read_excel('Energy Indicators.xls',
                           skiprows=17,
                           skipfooter=38)

energy = energy.drop(energy.columns[[0,1]], axis=1)

energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']

energy = energy.replace(to_replace={"Republic of Korea": "South Korea",
                                        "United States of America20": "United States",
                                        "United Kingdom of Great Britain and Northern Ireland19": "United Kingdom",
                                        "China, Hong Kong Special Administrative Region3": "Hong Kong",
                                        "Australia1":"Australia",
                                        "Bolivia (Plurinational State of)" : "Bolivia",
                                        "China2":"China",
                                        "China, Macao Special Administrative Region4":"China, Macao Special Administrative Region",
                                        "Denmark5":"Denmark",
                                        "Falkland Islands (Malvinas)": "Falkland Islands",
                                        "France6":"France",
                                        "Greenland7":"Greenland",
                                        "Indonesia8":"Indonesia",
                                        "Iran (Islamic Republic of)":"Iran",
                                        "Italy9":"Italy",
                                        "Japan10":"Japan",
                                        "Micronesia (Federated States of)":"Micronesia",
                                        "Netherlands12":"Netherlands",
                                        "Portugal13":"Portugal",
                                        "Saudi Arabia14":"Saudi Arabia",
                                        "Serbia15":"Serbia",
                                        "Sint Maarten (Dutch part)":"Sint Maarten",
                                        "Spain16":"Spain",
                                        "Switzerland17":"Switzerland",
                                        "Ukraine18":"Ukraine",
                                        "Venezuela (Bolivarian Republic of)":"Venezuela",
                                        "...": np.NaN})


energy["Energy Supply"] = energy["Energy Supply"] * 1000000

GDP = pd.read_csv('world_bank.csv', skiprows=4)

GDP = GDP.replace(to_replace={"Korea, Rep.": "South Korea", 
                                  "Iran, Islamic Rep.": "Iran",
                                  "Hong Kong SAR, China": "Hong Kong"})

ScimEn = pd.read_excel('scimagojr-3.xlsx')

GDP_mini = GDP.ix[:, ["Country Name","2006","2007","2008","2009","2010",
                          "2011","2012","2013","2014","2015"]]

merge1 = pd.merge(GDP_mini, ScimEn[:15], how='inner', left_on="Country Name",
                      right_on="Country")

del merge1["Country Name"]

merge2 = pd.merge(merge1, energy, how='inner', left_on="Country",
                      right_on="Country")
merge2 = merge2.set_index("Country")


def answer_one():
    
    return merge2

answer_one()

Top15 = answer_one()

#print(merge2['avgGDP'].sort_values(ascending=False))

def answer_three():
    avgGDP = Top15[['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']].mean(axis=1).rename('avgGDP').sort_values(ascending=False)
    print(type(avgGDP))

answer_three()



#print(Top15.where(Top15['Rank']==6)['2015'] - Top15.where(Top15['Rank']==6)['2006'])



print(Top15.iloc[2]['2015'] - Top15.iloc[2]['2006'])


print(Top15[Top15['Rank'] == 6]['2015'] - Top15[Top15['Rank'] == 6]['2006'])


Top15['avgGDP'] = Top15[['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']].mean(axis=1).sort_values(ascending=True)



Top15['GDPrank'] = Top15.avgGDP.rank(ascending=False)

print(np.mean(Top15['Energy Supply per Capita']))

Topcopy3 = Top15.copy()
Topcopy3['per_ren_country'] = list(zip(Topcopy3.index, Topcopy3['% Renewable']))

print(max(Topcopy3['per_ren_country'], key=lambda x: x[1]))



Topcopy5 = Top15.copy()
    
Topcopy5['pop_est'] = Topcopy5['Energy Supply'] / Topcopy5['Energy Supply per Capita']

Topcopy5['pop_rank'] = Topcopy5.pop_est.rank(ascending=False)


#print(Topcopy5.index.item([Topcopy5['pop_rank']==3]))

print(Topcopy5[Topcopy5['pop_rank']==3].index.tolist()[0])


Topcopy6 = Top15.copy()
    
Topcopy6['pop_est'] = Topcopy6['Energy Supply'] / Topcopy6['Energy Supply per Capita']
Topcopy6['cit_doc_per_capita'] = Topcopy6['Citable documents'] / Topcopy6['pop_est']
    

    


#print(Topcopy6['cit_doc_per_capita'].corr(Topcopy6['Energy Supply per Capita']))

Topcopy7 = Top15.copy()
    
median_ren = np.median(Topcopy7['% Renewable'])

print(median_ren)

Topcopy7['HighRenew'] = np.where(Topcopy7['% Renewable'] >= median_ren, 1, 0)

Topcopy7 = Topcopy7.sort_values(by='HighRenew', ascending=1)



Topcopy8 = Top15.copy()
Topcopy8['pop_est'] = Topcopy8['Energy Supply'] / Topcopy8['Energy Supply per Capita']  

ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
             
continents_df = pd.DataFrame()
continents_df['Country'] = ContinentDict.keys()
continents_df['Continent'] = ContinentDict.values()

continents_final = pd.DataFrame()
    
continents_2 = pd.merge(continents_df, Topcopy8, how='inner', left_on="Country",
                        right_index=True)


continents_final['Continent'] = ['Asia', 'Australia', 'Europe', 'North America', 'South America']
h = pd.DataFrame(continents_2.groupby(by=['Continent'])['pop_est'].sum())

continents_final = pd.merge(h, continents_final, how='inner', left_index=True, 
                        right_on='Continent')

continents_final['size'] = continents_final['pop_est']

del continents_final['pop_est']

h2 = pd.DataFrame(continents_2.groupby(by=['Continent'])['pop_est'].count())
h3 = pd.DataFrame(continents_2.groupby(by=['Continent'])['pop_est'].mean())
h4 = pd.DataFrame(continents_2.groupby(by=['Continent'])['pop_est'].std())

continents_final = pd.merge(h3, continents_final, how='inner', left_index=True, 
                        right_on='Continent')
continents_final['mean'] = continents_final['pop_est']
del continents_final['pop_est']
    
continents_final = pd.merge(h4, continents_final, how='inner', left_index=True, 
                        right_on='Continent')
continents_final['std'] = continents_final['pop_est']
del continents_final['pop_est']


Topcopy9 = Top15.copy()
    
Topcopy9['continent_cuts'] = pd.cut(Topcopy9['% Renewable'], bins=5)

#Topcopy9['Continents'] = ContinentDict.values()

Topcopy9['Continents'] = [ContinentDict[country] for country in Topcopy9.index]


Topcopy9 = Topcopy9.reset_index()


#print(Topcopy9.groupby(['Continents', 'continent_cuts']).size())


#print(Topcopy9.groupby(['Continents','continent_cuts']).size())


import locale
Top15 = answer_one()
Topcopy10 = Top15.copy()
Topcopy10['pop_est'] = Topcopy10['Energy Supply'] / Topcopy10['Energy Supply per Capita']
locale.setlocale(locale.LC_ALL, 'en_US.utf8')

new_num=list()
    
for i in Topcopy10['pop_est']:
    converted = locale.format("%d", i, grouping=True)
    new_num.append(converted)

country_list = Topcopy10.index.tolist()

print(country_list)


#PopEst = pd.Series(new_num)


money_tuples = list(zip(country_list, new_num))

PopEst = pd.DataFrame(money_tuples).set_index(0)[1]

PopEst.index.rename('Country')























{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_info = pd.read_csv('../DATA/lending_club_info.csv',index_col='LoanStatNew')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Description</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LoanStatNew</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>loan_amnt</th>\n",
       "      <td>The listed amount of the loan applied for by t...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>term</th>\n",
       "      <td>The number of payments on the loan. Values are...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>int_rate</th>\n",
       "      <td>Interest Rate on the loan</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>installment</th>\n",
       "      <td>The monthly payment owed by the borrower if th...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>grade</th>\n",
       "      <td>LC assigned loan grade</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sub_grade</th>\n",
       "      <td>LC assigned loan subgrade</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>emp_title</th>\n",
       "      <td>The job title supplied by the Borrower when ap...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>emp_length</th>\n",
       "      <td>Employment length in years. Possible values ar...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>home_ownership</th>\n",
       "      <td>The home ownership status provided by the borr...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>annual_inc</th>\n",
       "      <td>The self-reported annual income provided by th...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>verification_status</th>\n",
       "      <td>Indicates if income was verified by LC, not ve...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>issue_d</th>\n",
       "      <td>The month which the loan was funded</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>loan_status</th>\n",
       "      <td>Current status of the loan</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>purpose</th>\n",
       "      <td>A category provided by the borrower for the lo...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>title</th>\n",
       "      <td>The loan title provided by the borrower</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>zip_code</th>\n",
       "      <td>The first 3 numbers of the zip code provided b...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>addr_state</th>\n",
       "      <td>The state provided by the borrower in the loan...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>dti</th>\n",
       "      <td>A ratio calculated using the borrower’s total ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>earliest_cr_line</th>\n",
       "      <td>The month the borrower's earliest reported cre...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>open_acc</th>\n",
       "      <td>The number of open credit lines in the borrowe...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>pub_rec</th>\n",
       "      <td>Number of derogatory public records</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>revol_bal</th>\n",
       "      <td>Total credit revolving balance</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>revol_util</th>\n",
       "      <td>Revolving line utilization rate, or the amount...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>total_acc</th>\n",
       "      <td>The total number of credit lines currently in ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>initial_list_status</th>\n",
       "      <td>The initial listing status of the loan. Possib...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>application_type</th>\n",
       "      <td>Indicates whether the loan is an individual ap...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mort_acc</th>\n",
       "      <td>Number of mortgage accounts.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>pub_rec_bankruptcies</th>\n",
       "      <td>Number of public record bankruptcies</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                            Description\n",
       "LoanStatNew                                                            \n",
       "loan_amnt             The listed amount of the loan applied for by t...\n",
       "term                  The number of payments on the loan. Values are...\n",
       "int_rate                                      Interest Rate on the loan\n",
       "installment           The monthly payment owed by the borrower if th...\n",
       "grade                                            LC assigned loan grade\n",
       "sub_grade                                     LC assigned loan subgrade\n",
       "emp_title             The job title supplied by the Borrower when ap...\n",
       "emp_length            Employment length in years. Possible values ar...\n",
       "home_ownership        The home ownership status provided by the borr...\n",
       "annual_inc            The self-reported annual income provided by th...\n",
       "verification_status   Indicates if income was verified by LC, not ve...\n",
       "issue_d                             The month which the loan was funded\n",
       "loan_status                                  Current status of the loan\n",
       "purpose               A category provided by the borrower for the lo...\n",
       "title                           The loan title provided by the borrower\n",
       "zip_code              The first 3 numbers of the zip code provided b...\n",
       "addr_state            The state provided by the borrower in the loan...\n",
       "dti                   A ratio calculated using the borrower’s total ...\n",
       "earliest_cr_line      The month the borrower's earliest reported cre...\n",
       "open_acc              The number of open credit lines in the borrowe...\n",
       "pub_rec                             Number of derogatory public records\n",
       "revol_bal                                Total credit revolving balance\n",
       "revol_util            Revolving line utilization rate, or the amount...\n",
       "total_acc             The total number of credit lines currently in ...\n",
       "initial_list_status   The initial listing status of the loan. Possib...\n",
       "application_type      Indicates whether the loan is an individual ap...\n",
       "mort_acc                                   Number of mortgage accounts.\n",
       "pub_rec_bankruptcies               Number of public record bankruptcies"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_info"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.\n"
     ]
    }
   ],
   "source": [
    "print(data_info.loc['revol_util']['Description'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def feat_info(col_name):\n",
    "    print(data_info.loc[col_name]['Description'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of mortgage accounts.\n"
     ]
    }
   ],
   "source": [
    "feat_info('mort_acc')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('../DATA/lending_club_loan_two.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>loan_amnt</th>\n",
       "      <th>term</th>\n",
       "      <th>int_rate</th>\n",
       "      <th>installment</th>\n",
       "      <th>grade</th>\n",
       "      <th>sub_grade</th>\n",
       "      <th>emp_title</th>\n",
       "      <th>emp_length</th>\n",
       "      <th>home_ownership</th>\n",
       "      <th>annual_inc</th>\n",
       "      <th>...</th>\n",
       "      <th>open_acc</th>\n",
       "      <th>pub_rec</th>\n",
       "      <th>revol_bal</th>\n",
       "      <th>revol_util</th>\n",
       "      <th>total_acc</th>\n",
       "      <th>initial_list_status</th>\n",
       "      <th>application_type</th>\n",
       "      <th>mort_acc</th>\n",
       "      <th>pub_rec_bankruptcies</th>\n",
       "      <th>address</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>10000.0</td>\n",
       "      <td>36 months</td>\n",
       "      <td>11.44</td>\n",
       "      <td>329.48</td>\n",
       "      <td>B</td>\n",
       "      <td>B4</td>\n",
       "      <td>Marketing</td>\n",
       "      <td>10+ years</td>\n",
       "      <td>RENT</td>\n",
       "      <td>117000.0</td>\n",
       "      <td>...</td>\n",
       "      <td>16.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>36369.0</td>\n",
       "      <td>41.8</td>\n",
       "      <td>25.0</td>\n",
       "      <td>w</td>\n",
       "      <td>INDIVIDUAL</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0174 Michelle Gateway\\nMendozaberg, OK 22690</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8000.0</td>\n",
       "      <td>36 months</td>\n",
       "      <td>11.99</td>\n",
       "      <td>265.68</td>\n",
       "      <td>B</td>\n",
       "      <td>B5</td>\n",
       "      <td>Credit analyst</td>\n",
       "      <td>4 years</td>\n",
       "      <td>MORTGAGE</td>\n",
       "      <td>65000.0</td>\n",
       "      <td>...</td>\n",
       "      <td>17.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>20131.0</td>\n",
       "      <td>53.3</td>\n",
       "      <td>27.0</td>\n",
       "      <td>f</td>\n",
       "      <td>INDIVIDUAL</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1076 Carney Fort Apt. 347\\nLoganmouth, SD 05113</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>15600.0</td>\n",
       "      <td>36 months</td>\n",
       "      <td>10.49</td>\n",
       "      <td>506.97</td>\n",
       "      <td>B</td>\n",
       "      <td>B3</td>\n",
       "      <td>Statistician</td>\n",
       "      <td>&lt; 1 year</td>\n",
       "      <td>RENT</td>\n",
       "      <td>43057.0</td>\n",
       "      <td>...</td>\n",
       "      <td>13.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>11987.0</td>\n",
       "      <td>92.2</td>\n",
       "      <td>26.0</td>\n",
       "      <td>f</td>\n",
       "      <td>INDIVIDUAL</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>87025 Mark Dale Apt. 269\\nNew Sabrina, WV 05113</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7200.0</td>\n",
       "      <td>36 months</td>\n",
       "      <td>6.49</td>\n",
       "      <td>220.65</td>\n",
       "      <td>A</td>\n",
       "      <td>A2</td>\n",
       "      <td>Client Advocate</td>\n",
       "      <td>6 years</td>\n",
       "      <td>RENT</td>\n",
       "      <td>54000.0</td>\n",
       "      <td>...</td>\n",
       "      <td>6.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>5472.0</td>\n",
       "      <td>21.5</td>\n",
       "      <td>13.0</td>\n",
       "      <td>f</td>\n",
       "      <td>INDIVIDUAL</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>823 Reid Ford\\nDelacruzside, MA 00813</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>24375.0</td>\n",
       "      <td>60 months</td>\n",
       "      <td>17.27</td>\n",
       "      <td>609.33</td>\n",
       "      <td>C</td>\n",
       "      <td>C5</td>\n",
       "      <td>Destiny Management Inc.</td>\n",
       "      <td>9 years</td>\n",
       "      <td>MORTGAGE</td>\n",
       "      <td>55000.0</td>\n",
       "      <td>...</td>\n",
       "      <td>13.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>24584.0</td>\n",
       "      <td>69.8</td>\n",
       "      <td>43.0</td>\n",
       "      <td>f</td>\n",
       "      <td>INDIVIDUAL</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>679 Luna Roads\\nGreggshire, VA 11650</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 27 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   loan_amnt        term  int_rate  installment grade sub_grade  \\\n",
       "0    10000.0   36 months     11.44       329.48     B        B4   \n",
       "1     8000.0   36 months     11.99       265.68     B        B5   \n",
       "2    15600.0   36 months     10.49       506.97     B        B3   \n",
       "3     7200.0   36 months      6.49       220.65     A        A2   \n",
       "4    24375.0   60 months     17.27       609.33     C        C5   \n",
       "\n",
       "                 emp_title emp_length home_ownership  annual_inc  ...  \\\n",
       "0                Marketing  10+ years           RENT    117000.0  ...   \n",
       "1          Credit analyst     4 years       MORTGAGE     65000.0  ...   \n",
       "2             Statistician   < 1 year           RENT     43057.0  ...   \n",
       "3          Client Advocate    6 years           RENT     54000.0  ...   \n",
       "4  Destiny Management Inc.    9 years       MORTGAGE     55000.0  ...   \n",
       "\n",
       "  open_acc pub_rec revol_bal revol_util total_acc  initial_list_status  \\\n",
       "0     16.0     0.0   36369.0       41.8      25.0                    w   \n",
       "1     17.0     0.0   20131.0       53.3      27.0                    f   \n",
       "2     13.0     0.0   11987.0       92.2      26.0                    f   \n",
       "3      6.0     0.0    5472.0       21.5      13.0                    f   \n",
       "4     13.0     0.0   24584.0       69.8      43.0                    f   \n",
       "\n",
       "  application_type  mort_acc  pub_rec_bankruptcies  \\\n",
       "0       INDIVIDUAL       0.0                   0.0   \n",
       "1       INDIVIDUAL       3.0                   0.0   \n",
       "2       INDIVIDUAL       0.0                   0.0   \n",
       "3       INDIVIDUAL       0.0                   0.0   \n",
       "4       INDIVIDUAL       1.0                   0.0   \n",
       "\n",
       "                                           address  \n",
       "0     0174 Michelle Gateway\\nMendozaberg, OK 22690  \n",
       "1  1076 Carney Fort Apt. 347\\nLoganmouth, SD 05113  \n",
       "2  87025 Mark Dale Apt. 269\\nNew Sabrina, WV 05113  \n",
       "3            823 Reid Ford\\nDelacruzside, MA 00813  \n",
       "4             679 Luna Roads\\nGreggshire, VA 11650  \n",
       "\n",
       "[5 rows x 27 columns]"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 396030 entries, 0 to 396029\n",
      "Data columns (total 27 columns):\n",
      " #   Column                Non-Null Count   Dtype  \n",
      "---  ------                --------------   -----  \n",
      " 0   loan_amnt             396030 non-null  float64\n",
      " 1   term                  396030 non-null  object \n",
      " 2   int_rate              396030 non-null  float64\n",
      " 3   installment           396030 non-null  float64\n",
      " 4   grade                 396030 non-null  object \n",
      " 5   sub_grade             396030 non-null  object \n",
      " 6   emp_title             373103 non-null  object \n",
      " 7   emp_length            377729 non-null  object \n",
      " 8   home_ownership        396030 non-null  object \n",
      " 9   annual_inc            396030 non-null  float64\n",
      " 10  verification_status   396030 non-null  object \n",
      " 11  issue_d               396030 non-null  object \n",
      " 12  loan_status           396030 non-null  object \n",
      " 13  purpose               396030 non-null  object \n",
      " 14  title                 394275 non-null  object \n",
      " 15  dti                   396030 non-null  float64\n",
      " 16  earliest_cr_line      396030 non-null  object \n",
      " 17  open_acc              396030 non-null  float64\n",
      " 18  pub_rec               396030 non-null  float64\n",
      " 19  revol_bal             396030 non-null  float64\n",
      " 20  revol_util            395754 non-null  float64\n",
      " 21  total_acc             396030 non-null  float64\n",
      " 22  initial_list_status   396030 non-null  object \n",
      " 23  application_type      396030 non-null  object \n",
      " 24  mort_acc              358235 non-null  float64\n",
      " 25  pub_rec_bankruptcies  395495 non-null  float64\n",
      " 26  address               396030 non-null  object \n",
      "dtypes: float64(12), object(15)\n",
      "memory usage: 81.6+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29c7d10b490>"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='loan_status',data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29c00b5cf70>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12,4))\n",
    "sns.distplot(df['loan_amnt'],kde=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',\n",
       "       'emp_title', 'emp_length', 'home_ownership', 'annual_inc',\n",
       "       'verification_status', 'issue_d', 'loan_status', 'purpose', 'title',\n",
       "       'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal',\n",
       "       'revol_util', 'total_acc', 'initial_list_status', 'application_type',\n",
       "       'mort_acc', 'pub_rec_bankruptcies', 'address'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>loan_amnt</th>\n",
       "      <th>int_rate</th>\n",
       "      <th>installment</th>\n",
       "      <th>annual_inc</th>\n",
       "      <th>dti</th>\n",
       "      <th>open_acc</th>\n",
       "      <th>pub_rec</th>\n",
       "      <th>revol_bal</th>\n",
       "      <th>revol_util</th>\n",
       "      <th>total_acc</th>\n",
       "      <th>mort_acc</th>\n",
       "      <th>pub_rec_bankruptcies</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>loan_amnt</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.168921</td>\n",
       "      <td>0.953929</td>\n",
       "      <td>0.336887</td>\n",
       "      <td>0.016636</td>\n",
       "      <td>0.198556</td>\n",
       "      <td>-0.077779</td>\n",
       "      <td>0.328320</td>\n",
       "      <td>0.099911</td>\n",
       "      <td>0.223886</td>\n",
       "      <td>0.222315</td>\n",
       "      <td>-0.106539</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>int_rate</th>\n",
       "      <td>0.168921</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.162758</td>\n",
       "      <td>-0.056771</td>\n",
       "      <td>0.079038</td>\n",
       "      <td>0.011649</td>\n",
       "      <td>0.060986</td>\n",
       "      <td>-0.011280</td>\n",
       "      <td>0.293659</td>\n",
       "      <td>-0.036404</td>\n",
       "      <td>-0.082583</td>\n",
       "      <td>0.057450</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>installment</th>\n",
       "      <td>0.953929</td>\n",
       "      <td>0.162758</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.330381</td>\n",
       "      <td>0.015786</td>\n",
       "      <td>0.188973</td>\n",
       "      <td>-0.067892</td>\n",
       "      <td>0.316455</td>\n",
       "      <td>0.123915</td>\n",
       "      <td>0.202430</td>\n",
       "      <td>0.193694</td>\n",
       "      <td>-0.098628</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>annual_inc</th>\n",
       "      <td>0.336887</td>\n",
       "      <td>-0.056771</td>\n",
       "      <td>0.330381</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.081685</td>\n",
       "      <td>0.136150</td>\n",
       "      <td>-0.013720</td>\n",
       "      <td>0.299773</td>\n",
       "      <td>0.027871</td>\n",
       "      <td>0.193023</td>\n",
       "      <td>0.236320</td>\n",
       "      <td>-0.050162</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>dti</th>\n",
       "      <td>0.016636</td>\n",
       "      <td>0.079038</td>\n",
       "      <td>0.015786</td>\n",
       "      <td>-0.081685</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.136181</td>\n",
       "      <td>-0.017639</td>\n",
       "      <td>0.063571</td>\n",
       "      <td>0.088375</td>\n",
       "      <td>0.102128</td>\n",
       "      <td>-0.025439</td>\n",
       "      <td>-0.014558</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>open_acc</th>\n",
       "      <td>0.198556</td>\n",
       "      <td>0.011649</td>\n",
       "      <td>0.188973</td>\n",
       "      <td>0.136150</td>\n",
       "      <td>0.136181</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.018392</td>\n",
       "      <td>0.221192</td>\n",
       "      <td>-0.131420</td>\n",
       "      <td>0.680728</td>\n",
       "      <td>0.109205</td>\n",
       "      <td>-0.027732</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>pub_rec</th>\n",
       "      <td>-0.077779</td>\n",
       "      <td>0.060986</td>\n",
       "      <td>-0.067892</td>\n",
       "      <td>-0.013720</td>\n",
       "      <td>-0.017639</td>\n",
       "      <td>-0.018392</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.101664</td>\n",
       "      <td>-0.075910</td>\n",
       "      <td>0.019723</td>\n",
       "      <td>0.011552</td>\n",
       "      <td>0.699408</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>revol_bal</th>\n",
       "      <td>0.328320</td>\n",
       "      <td>-0.011280</td>\n",
       "      <td>0.316455</td>\n",
       "      <td>0.299773</td>\n",
       "      <td>0.063571</td>\n",
       "      <td>0.221192</td>\n",
       "      <td>-0.101664</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.226346</td>\n",
       "      <td>0.191616</td>\n",
       "      <td>0.194925</td>\n",
       "      <td>-0.124532</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>revol_util</th>\n",
       "      <td>0.099911</td>\n",
       "      <td>0.293659</td>\n",
       "      <td>0.123915</td>\n",
       "      <td>0.027871</td>\n",
       "      <td>0.088375</td>\n",
       "      <td>-0.131420</td>\n",
       "      <td>-0.075910</td>\n",
       "      <td>0.226346</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.104273</td>\n",
       "      <td>0.007514</td>\n",
       "      <td>-0.086751</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>total_acc</th>\n",
       "      <td>0.223886</td>\n",
       "      <td>-0.036404</td>\n",
       "      <td>0.202430</td>\n",
       "      <td>0.193023</td>\n",
       "      <td>0.102128</td>\n",
       "      <td>0.680728</td>\n",
       "      <td>0.019723</td>\n",
       "      <td>0.191616</td>\n",
       "      <td>-0.104273</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.381072</td>\n",
       "      <td>0.042035</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mort_acc</th>\n",
       "      <td>0.222315</td>\n",
       "      <td>-0.082583</td>\n",
       "      <td>0.193694</td>\n",
       "      <td>0.236320</td>\n",
       "      <td>-0.025439</td>\n",
       "      <td>0.109205</td>\n",
       "      <td>0.011552</td>\n",
       "      <td>0.194925</td>\n",
       "      <td>0.007514</td>\n",
       "      <td>0.381072</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.027239</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>pub_rec_bankruptcies</th>\n",
       "      <td>-0.106539</td>\n",
       "      <td>0.057450</td>\n",
       "      <td>-0.098628</td>\n",
       "      <td>-0.050162</td>\n",
       "      <td>-0.014558</td>\n",
       "      <td>-0.027732</td>\n",
       "      <td>0.699408</td>\n",
       "      <td>-0.124532</td>\n",
       "      <td>-0.086751</td>\n",
       "      <td>0.042035</td>\n",
       "      <td>0.027239</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      loan_amnt  int_rate  installment  annual_inc       dti  \\\n",
       "loan_amnt              1.000000  0.168921     0.953929    0.336887  0.016636   \n",
       "int_rate               0.168921  1.000000     0.162758   -0.056771  0.079038   \n",
       "installment            0.953929  0.162758     1.000000    0.330381  0.015786   \n",
       "annual_inc             0.336887 -0.056771     0.330381    1.000000 -0.081685   \n",
       "dti                    0.016636  0.079038     0.015786   -0.081685  1.000000   \n",
       "open_acc               0.198556  0.011649     0.188973    0.136150  0.136181   \n",
       "pub_rec               -0.077779  0.060986    -0.067892   -0.013720 -0.017639   \n",
       "revol_bal              0.328320 -0.011280     0.316455    0.299773  0.063571   \n",
       "revol_util             0.099911  0.293659     0.123915    0.027871  0.088375   \n",
       "total_acc              0.223886 -0.036404     0.202430    0.193023  0.102128   \n",
       "mort_acc               0.222315 -0.082583     0.193694    0.236320 -0.025439   \n",
       "pub_rec_bankruptcies  -0.106539  0.057450    -0.098628   -0.050162 -0.014558   \n",
       "\n",
       "                      open_acc   pub_rec  revol_bal  revol_util  total_acc  \\\n",
       "loan_amnt             0.198556 -0.077779   0.328320    0.099911   0.223886   \n",
       "int_rate              0.011649  0.060986  -0.011280    0.293659  -0.036404   \n",
       "installment           0.188973 -0.067892   0.316455    0.123915   0.202430   \n",
       "annual_inc            0.136150 -0.013720   0.299773    0.027871   0.193023   \n",
       "dti                   0.136181 -0.017639   0.063571    0.088375   0.102128   \n",
       "open_acc              1.000000 -0.018392   0.221192   -0.131420   0.680728   \n",
       "pub_rec              -0.018392  1.000000  -0.101664   -0.075910   0.019723   \n",
       "revol_bal             0.221192 -0.101664   1.000000    0.226346   0.191616   \n",
       "revol_util           -0.131420 -0.075910   0.226346    1.000000  -0.104273   \n",
       "total_acc             0.680728  0.019723   0.191616   -0.104273   1.000000   \n",
       "mort_acc              0.109205  0.011552   0.194925    0.007514   0.381072   \n",
       "pub_rec_bankruptcies -0.027732  0.699408  -0.124532   -0.086751   0.042035   \n",
       "\n",
       "                      mort_acc  pub_rec_bankruptcies  \n",
       "loan_amnt             0.222315             -0.106539  \n",
       "int_rate             -0.082583              0.057450  \n",
       "installment           0.193694             -0.098628  \n",
       "annual_inc            0.236320             -0.050162  \n",
       "dti                  -0.025439             -0.014558  \n",
       "open_acc              0.109205             -0.027732  \n",
       "pub_rec               0.011552              0.699408  \n",
       "revol_bal             0.194925             -0.124532  \n",
       "revol_util            0.007514             -0.086751  \n",
       "total_acc             0.381072              0.042035  \n",
       "mort_acc              1.000000              0.027239  \n",
       "pub_rec_bankruptcies  0.027239              1.000000  "
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.corr()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29c00bc0fd0>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12,6))\n",
    "sns.heatmap(df.corr(),annot=True,cmap='viridis')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.\n"
     ]
    }
   ],
   "source": [
    "feat_info('loan_amnt')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The monthly payment owed by the borrower if the loan originates.\n"
     ]
    }
   ],
   "source": [
    "feat_info('installment')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29c010cd760>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.scatterplot(x='installment',y='loan_amnt',data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29c010e5f10>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZIAAAEHCAYAAACEKcAKAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3df5xVdb3v8debwRQzVHAk7oBBDZ1SMsrRS9k94Y+SYyp60hM9ugfscqPjNaJzu5XaOSfrPPCRj255hNRzMb2iWUqUiR6lVDQ7NwIHIxF/5BQovwJERfzFkfFz/1jfkTXDZmY2a/bsmdnv5+OxH3utz/5+1/4u3M5nf9d37e9XEYGZmdn+GlTtBpiZWf/mRGJmZoU4kZiZWSFOJGZmVogTiZmZFTK42g2ohiOOOCLGjBlT7WaYmfUrK1eufDYi6jvGazKRjBkzhubm5mo3w8ysX5H0dKm4L22ZmVkhTiRmZlaIE4mZmRXiRGJmZoU4kVghkyZNevNh1pfMnTuXSZMmcdVVV1W7KQNeryQSSXWSfifpzrQ/TNI9kp5Kz4fnyl4sqUXSk5JOy8WPk7Q6vTZXklL8QEm3pvhySWN645zMrG/72c9+BsBPfvKTKrdk4OutHsls4PHc/kXAfRExDrgv7SPpaGAqcAwwGbhaUl2qcw0wExiXHpNTfAbwfEQ0AlcAl1f2VKxNx16IeyXWV8ydO7fdvnsllVXxRCJpFPAJ4Ae58BRgQdpeAJydi98SEbsiYi3QApwgaSQwNCKWRTbv/Y0d6rQdaxFwSltvxcxqU1tvpI17JZXVGz2SfwG+CryRi42IiM0A6fnIFG8A1ufKbUixhrTdMd6uTkTsBnYAwzs2QtJMSc2Smrdt21b0nMzMLKloIpF0BrA1IlZ2t0qJWHQS76xO+0DE/Ihoioim+vq9fuFvZmb7qdJTpJwInCXpdOAgYKikHwJbJI2MiM3pstXWVH4DMDpXfxSwKcVHlYjn62yQNBg4FHiuUidkZn3foEGDeOONN9rtW+VU9F83Ii6OiFERMYZsEH1pRPxXYDEwPRWbDtyethcDU9OdWGPJBtVXpMtfOyVNTOMf0zrUaTvWuek9vH6wWQ0744wz2u2feeaZVWpJbahWmv428DFJTwEfS/tExBpgIfAYsAS4MCJaU50LyAbsW4A/Anen+HXAcEktwP8k3QFmldfxngbf42B9xfTp09vtT5s2rUotqQ29NvtvRDwAPJC2twOn7KPcHGBOiXgzML5E/DXgvB5sqnVTfX09W7dufXP/yCOP7KS0We8ZPnw4Z511FnfccQdnnXUWw4fvdf+N9aCanEbeesb27dvb7T/77LNVaonZ3qZPn866devcG+kFTiS23/KDmaX2zapp+PDhe/0w0SrDtzLYfut4T4PvcTCrTU4kZjYgbd++nS9+8Yt7XYK1nudEYmYD0oIFC1i9ejU33nhjtZsy4DmRmNmAs337dpYsWUJEsGTJEvdKKsyJxPZbXV1dp/tm1bJgwYI3b/5obW11r6TCnEhsv7W2tna6b1Yt9957L7t37wZg9+7d3HPPPVVu0cDmRGJmA86pp57K4MHZrxsGDx7Mxz72sSq3aGDz70j6qXnz5tHS0lLtZuxl9uzZVXnfxsZGZs2aVZX3tr5n+vTpLFmyBMguufpHiZXlHomZDTjDhw9n8uTJSGLy5MmeIqXCVIs/Imtqaorm5uZqN6PfW7p0Kd/61rfe3P/GN77BSSedVMUWWV/RF3rM69ev58UXX+Td7343BxxwQFXbMlB6zJJWRkRTx7h7JLbfTj755De3Bw8e7CRifcquXbs48MADq55EaoHHSKyQ0aNHs379er7+9a9XuynWh/SFb99t43VXXnlllVsy8DmRWCHDhg1j2LBh7o2Y1TBf2jIzs0IqmkgkHSRphaTfS1oj6ZspfqmkjZJWpcfpuToXS2qR9KSk03Lx4yStTq/NTUvukpblvTXFl0saU8lzMjOz9irdI9kFnBwR7wcmAJMlTUyvXRERE9LjLgBJR5Ot7X4MMBm4WlLbvBvXADPJ1nEfl14HmAE8HxGNwBXA5RU+JzMzy6loIonMS2n3gPTo7H7jKcAtEbErItaSrc9+gqSRwNCIWBbZ/co3Amfn6ixI24uAU9p6K2ZmVnkVHyORVCdpFbAVuCcilqeXviDpEUnXSzo8xRqA9bnqG1KsIW13jLerExG7gR3AXr8+kjRTUrOk5m3btvXQ2ZmZWcUTSUS0RsQEYBRZ72I82WWqd5Fd7toMfDcVL9WTiE7indXp2I75EdEUEU319fVlnoWZme1Lr921FREvAA8AkyNiS0owbwDXAiekYhuA0blqo4BNKT6qRLxdHUmDgUOB5yp0GmZm1kGl79qql3RY2h4CnAo8kcY82pwDPJq2FwNT051YY8kG1VdExGZgp6SJafxjGnB7rs70tH0usDRqcd4XM7MqqfQPEkcCC9KdV4OAhRFxp6SbJE0guwS1Dvg8QESskbQQeAzYDVwYEW2LXFwA3AAMAe5OD4DrgJsktZD1RKZW+JzMzCynookkIh4BPlAi/red1JkDzCkRbwbGl4i/BpxXrKVmZra//Mt2MzMrxInEzMwKcSIxM7NCnEjMzKwQJxIzMyvEicTMzApxIjEzs0KcSMzMrBAnEjMzK8SJxMzMCnEiMTOzQpxIzMysECcSMzMrxInEzMwKcSIxM7NCnEjMzKyQSi+1e5CkFZJ+L2mNpG+m+DBJ90h6Kj0fnqtzsaQWSU9KOi0XP07S6vTa3LTkLmlZ3ltTfLmkMZU8JzMza6/SPZJdwMkR8X5gAjBZ0kTgIuC+iBgH3Jf2kXQ02VK5xwCTgavTMr0A1wAzydZxH5deB5gBPB8RjcAVwOUVPiczM8upaCKJzEtp94D0CGAKsCDFFwBnp+0pwC0RsSsi1gItwAmSRgJDI2JZRARwY4c6bcdaBJzS1lsxM7PKq/gYiaQ6SauArcA9EbEcGBERmwHS85GpeAOwPld9Q4o1pO2O8XZ1ImI3sAMYXqIdMyU1S2retm1bT52emVnNq3giiYjWiJgAjCLrXYzvpHipnkR0Eu+sTsd2zI+Ipohoqq+v76rZZmbWTb1211ZEvAA8QDa2sSVdriI9b03FNgCjc9VGAZtSfFSJeLs6kgYDhwLPVeQkzMxsL5W+a6te0mFpewhwKvAEsBiYnopNB25P24uBqelOrLFkg+or0uWvnZImpvGPaR3qtB3rXGBpGkcxM7NeMLjCxx8JLEh3Xg0CFkbEnZKWAQslzQCeAc4DiIg1khYCjwG7gQsjojUd6wLgBmAIcHd6AFwH3CSphawnMrXC52RmZjkVTSQR8QjwgRLx7cAp+6gzB5hTIt4M7DW+EhGvkRKRmZn1Pv+y3czMCnEiMTOzQpxIzMysECcSMzMrxInEzMwKcSIxM7NCnEjMzKwQJxIzMyvEicTMzApxIjEzs0KcSMzMrBAnEjMzK8SJxMzMCnEiMTOzQpxIzMysECcSMzMrpNJL7Y6WdL+kxyWtkTQ7xS+VtFHSqvQ4PVfnYkktkp6UdFoufpyk1em1uWnJXdKyvLem+HJJYyp5TmZm1l6leyS7gS9HxHuBicCFko5Or10RERPS4y6A9NpU4BhgMnB1WqYX4BpgJtk67uPS6wAzgOcjohG4Ari8wudkZmY5FU0kEbE5Ih5O2zuBx4GGTqpMAW6JiF0RsRZoAU6QNBIYGhHLIiKAG4Gzc3UWpO1FwCltvRUzM6u8bicSSWO7E+uk/hiy9duXp9AXJD0i6XpJh6dYA7A+V21DijWk7Y7xdnUiYjewAxhe4v1nSmqW1Lxt27buNtvMzLpQTo/kpyVii7pTUdIhqf6XIuJFsstU7wImAJuB77YVLVE9Ool3Vqd9IGJ+RDRFRFN9fX13mm1mZt0wuKsCkt5DNmZxqKS/zr00FDioG/UPIEsiN0fEzwAiYkvu9WuBO9PuBmB0rvooYFOKjyoRz9fZIGkwcCjwXFftMjOzntGdHslfAGcAhwFn5h4fBD7XWcU0VnEd8HhEfC8XH5krdg7waNpeDExNd2KNJRtUXxERm4GdkiamY04Dbs/VmZ62zwWWpnEUMzPrBV32SCLiduB2SR+KiGVlHv9E4G+B1ZJWpdglwKclTSC7BLUO+Hx6rzWSFgKPkd3xdWFEtKZ6FwA3AEOAu9MDskR1k6QWsp7I1DLbaGZmBXSZSHJaJF0CjMnXi4j/tq8KEfHvlB7DuKuTOnOAOSXizcD4EvHXgPM6a7iZmVVOOYnkduDXwL1AaxdlzcysRpSTSA6OiK9VrCVmZtYvlXP77535qUzMzMygvEQymyyZvCrpRUk7Jb1YqYaZmVn/0O1LWxHxtko2xMzM+qdyxkiQdCx737X1sx5uk5mZ9SPdTiSSrgeOBdYAb6RwAE4kZmY1rJweycSIOLrrYmZmVkvKGWxflltLxMzMDCivR7KALJn8GdhF9ov1iIhjK9IyMzPrF8pJJNeT5s1izxiJmZnVuHISyTMRsbhiLTEzs36pnETyhKQfAXeQXdoCfPuvmVmtKyeRDCFLIB/PxXz7r5lZjSvnl+2frWRDzMysfyrnB4kHATPIlt19c4ndztYjMTOzga+c35HcBLwdOA34Fdm66Ts7qyBptKT7JT0uaY2k2Sk+TNI9kp5Kz4fn6lwsqUXSk5JOy8WPk7Q6vTY3LblLWpb31hRfLmlMGedkZmYFlZNIGiPiH4GXI2IB8AngfV3U2Q18OSLeC0wELkw/arwIuC8ixgH3pX3Sa1PJej2Tgasl1aVjXQPMJFvHfVx6HbJe0vMR0QhcAVxexjmZmVlB5Qy2v56eX5A0Hvgz2QSO+xQRm4HNaXunpMeBBmAKMCkVWwA8AHwtxW+JiF3A2rQO+wmS1gFD29aMl3QjcDbZuu1TgEvTsRYB35ekiIgyzs1sQJg3bx4tLS3Vbkaf0PbvMHv27Cq3pG9obGxk1qxZFTl2OYlkfroE9Q/AYuAQ4B+7WzldcvoAsBwYkZIMEbFZ0pGpWAPw21y1DSn2etruGG+rsz4da7ekHcBw4NkO7z+TrEfDUUcd1d1mm/UrLS0tPLXmdxx1iFfDfsvr2QWXXU83V7kl1ffMS3VdFyqgnLu2fpA2HwTe2fF1SdPTJa+9SDoE+CnwpYh4MQ1vlCxa6q07iXdWp30gYj4wH6Cpqcm9FRuwjjqklUs+6DXnbI/LHh5a0eOXM0bSlZL9R0kHkCWRm3M/XtwiaWR6fSSwNcU3AKNz1UcBm1J8VIl4uzqSBgOHAs8VPRkzM+uesha26sJePYN0Z9V1wOMR8b3cS4uB6cC30/PtufiPJH0P+E9kg+orIqI1Le07kezS2DRgXodjLQPOBZZWenzE16H38HXo9ip5Hdqsr+rJRFLqj/eJpIkeJa1KsUvIEshCSTOAZ4DzACJijaSFwGNkd3xdGBFtF3svAG4g+4X93ekBWaK6KQ3MP0d211dFtbS0sOrRx2k9eFil36rPG/Qf2X/2lX/aUuWWVF/dK+4IW22qaI8kIv69VDw5pVQwIuYAc0rEm4HxJeKvkRJRb2o9eBivvuf03n5b68OGPHFXtZtgVhU9OUby/3rwWGZm1k+UM0XKgcAnyX478ma9iPhWev5CTzfOzMz6vnIubd0O7ABWkptG3szMals5iWRUREzuupiZmdWScsZIfiOpq7m1zMysxpTTI/kIcL6ktWSXtgRERBxbkZaZmVm/UE4i+auKtcLMzPqtcubaehogTbB4UBfFzcysRnR7jETSWZKeAtaSLWy1jj2/LjczsxpVzmD7P5MtTvWHiBhL9st0/wjRzKzGlZNIXo+I7cAgSYMi4n5gQoXaZWZm/UQ5g+0vpHVFfg3cLGkr2cSKZmZWw8rpkUwBXgW+BCwB/gicWYlGmZlZ/1HOXVsvSxoBHA9sB+5Ol7rMzKyGlXPX1t8AK8imbP8bYLmkcyvVMDMz6x/KGSP5OnB8RGwFkFQP3AssqkTDzMysfyhnjGRQWxJJtndVX9L1krZKejQXu1TSRkmr0uP03GsXS2qR9KSk03Lx4yStTq/NTUv4IulASbem+HJJY8o4HzMz6wHlJJIlkn4h6XxJ5wP/BnS1JNwNQKkZg6+IiAnpcReApKPJlsk9JtW5WlJdKn8NMJNsDfdxuWPOAJ6PiEbgCuDyMs7HzMx6QLcTSUR8BZgPHAu8H5gfEV/ros6DZOuod8cU4JaI2BURa4EW4ARJI4GhEbEsIgK4ETg7V2dB2l4EnNLWWzEzs95R1prtEfFT4Kc98L5fkDQNaAa+HBHPAw3Ab3NlNqTY62m7Y5z0vD61bbekHcBw4NmObyhpJlmvhqOOOqoHTsHMzKAbPRJJOyW9WOKxU9KL+/Ge1wDvIvtV/Gbgu21vVaJsdBLvrM7ewYj5EdEUEU319fXltdjMzPapyx5JRLytJ98wIra0bUu6Frgz7W4ARueKjgI2pfioEvF8nQ2SBgOH0v1LaWZm1gPKGWzvEWnMo805QNsdXYuBqelOrLFkg+orImIzsFPSxDT+MY1s/fi2OtPT9rnA0jSOYmZmvaSsMZJySfoxMAk4QtIG4BvAJEkTyC5BrQM+DxARayQtBB4jm8PrwohoTYe6gOwOsCFkU9e3TV9/HXCTpBaynsjUSp6PWV+3ceNGXt5Zx2UPD612U6wPeXpnHW/duLFix69oIomIT5cIX9dJ+TnAnBLxZmB8ifhrZL+0NzOzKqloIjGz3tXQ0MCu3Zu55IP7cx+MDVSXPTyUAxsaui64n3p9jMTMzAYWJxIzMyvEicTMzApxIjEzs0KcSMzMrBAnEjMzK8SJxMzMCvHvSPbDxo0bqXtlB0Oe6Go5Fqslda9sZ+PG3dVuhlmvc4/EzMwKcY9kPzQ0NPDnXYN59T2nd13YasaQJ+6ioWFEtZth1uvcIzEzs0KcSMzMrBAnEjMzK8SJxMzMCnEiMTOzQiqaSCRdL2mrpEdzsWGS7pH0VHo+PPfaxZJaJD0p6bRc/DhJq9Nrc9OSu6RleW9N8eWSxlTyfMzMbG+V7pHcAEzuELsIuC8ixgH3pX0kHU22VO4xqc7VkupSnWuAmWTruI/LHXMG8HxENAJXAJdX7EzMzKykiiaSiHiQbC31vCnAgrS9ADg7F78lInZFxFqgBThB0khgaEQsi4gAbuxQp+1Yi4BT2norZmbWO6oxRjIiIjYDpOcjU7wBWJ8rtyHFGtJ2x3i7OhGxG9gBDC/1ppJmSmqW1Lxt27YeOhUzM+tLg+2lehLRSbyzOnsHI+ZHRFNENNXX1+9nE83MrKNqJJIt6XIV6Xlrim8ARufKjQI2pfioEvF2dSQNBg5l70tpZmZWQdVIJIuB6Wl7OnB7Lj413Yk1lmxQfUW6/LVT0sQ0/jGtQ522Y50LLE3jKGZm1ksqOmmjpB8Dk4AjJG0AvgF8G1goaQbwDHAeQESskbQQeAzYDVwYEa3pUBeQ3QE2BLg7PQCuA26S1ELWE5layfMxM7O9VTSRRMSn9/HSKfsoPweYUyLeDIwvEX+NlIjMzKw6+tJgu5mZ9UNej8RsgHnmpToue3hotZtRdVteyb4njzj4jSq3pPqeeamOcRU8vhOJ2QDS2NhY7Sb0Gf/R0gLAge/wv8k4KvvZcCIxG0BmzZpV7Sb0GbNnzwbgyiuvrHJLBj6PkZiZWSFOJGZmVogvbe2nuleeY8gTd1W7GVU36LUXAXjjIA/u1r3yHDCi2s0w63VOJPvBA5p7tLTsBKDxnf4DCiP82bCa5ESyHzyguYcHNM3MYyRmZlaIE4mZmRXiRGJmZoU4kZiZWSFOJGZmVogTiZmZFeJEYmZmhVQtkUhaJ2m1pFWSmlNsmKR7JD2Vng/Plb9YUoukJyWdlosfl47TImluWo7XzMx6SbV7JCdFxISIaEr7FwH3RcQ44L60j6SjyZbRPQaYDFwtqS7VuQaYSTZT8rj0upmZ9ZJqJ5KOpgAL0vYC4Oxc/JaI2BURa4EW4ARJI4GhEbEsIgK4MVfHzMx6QTUTSQC/lLRS0swUGxERmwHS85Ep3gCsz9XdkGINabtjfC+SZkpqltS8bdu2HjwNM7PaVs25tk6MiE2SjgTukfREJ2VLjXtEJ/G9gxHzgfkATU1NJcuYmVn5qtYjiYhN6XkrcBtwArAlXa4iPW9NxTcAo3PVRwGbUnxUibiZmfWSqiQSSW+V9La2beDjwKPAYmB6KjYduD1tLwamSjpQ0liyQfUV6fLXTkkT091a03J1zMysF1Tr0tYI4LZ0p+5g4EcRsUTSQ8BCSTOAZ4DzACJijaSFwGPAbuDCiGhNx7oAuAEYAtydHmZm1kuqkkgi4k/A+0vEtwOn7KPOHGBOiXgzML6n22hmZt3T127/NTOzfsaJxMzMCnEiMTOzQpxIzMysECcSMzMrxInEzMwKcSIxM7NCnEjMzKwQJxIzMyvEicTMzApxIjEzs0KcSMzMrBAnEjMzK8SJxMzMCqnmUrtmNkDNmzePlpaWqrah7f1nz55d1XYANDY2MmvWrGo3o2KcSMxsQBoyZEi1m1AzFBHVbkNhkiYDVwJ1wA8i4tudlW9qaorm5uZeaVul9IVvfLDnW19jY2NV2zHQv/GZ9QWSVkZEU8d4vx8jkVQHXAX8FXA08GlJR1e3VbVjyJAh/uZnVuMGwqWtE4CWtHwvkm4BppCt7z5g+du3mfUV/b5HAjQA63P7G1LMzMx6wUBIJCoR22vgR9JMSc2Smrdt29YLzTIzqw0DIZFsAEbn9kcBmzoWioj5EdEUEU319fW91jgzs4FuICSSh4BxksZKegswFVhc5TaZmdWMfj/YHhG7JX0B+AXZ7b/XR8SaKjfLzKxm9PtEAhARdwF3VbsdZma1aCBc2jIzsypyIjEzs0IGxBQp5ZK0DXi62u0YQI4Anq12I8xK8GezZ70jIva67bUmE4n1LEnNpebfMas2fzZ7hy9tmZlZIU4kZmZWiBOJ9YT51W6A2T74s9kLPEZiZmaFuEdiZmaFOJGYmVkhTiQ1QlKrpFW5x5hOyp4v6ftp+1JJ/6uM97lB0tr0Hg9L+lAX5X/TyXHO7e77Wt8n6e2SbpH0R0mPSbpL0rslTZJ0Z5XbVvLzpsw/SHpK0h8k3S/pmNzr50l6XNL9af/Hkh6R9Pe92f5qGxBzbVm3vBoRE3rpvb4SEYskfRz4P8Cx+yoYER/upTZZFUkScBuwICKmptgEYEQPHHtwROwuepx9uBD4MPD+iHglfaYXSzomIl4DZgD/IyLul/R24MMR8Y4KtaXPco+khklaJ+mItN0k6YFOyr5L0sO5/XGSVnbxFg8CjZIOkXRf6qGsljQld5yX0rMkfT99U/034Mgi52Z9zknA6xHxr22BiFgVEb9Ou4dIWiTpCUk3p8SDpH+S9JCkRyXNz8UfkHSZpF8BsyUdn3oCyyR9R9KjqVxd2n8ovf75FO/u5+1rwKyIeCW1+ZfAb4DPSPon4CPAv0r6DvBL4MjUG/8vPfvP17c5kdSOIbnLWreVWzki/gjsSN8iAT4L3NBFtTOB1cBrwDkR8UGyPyjfbfuDkHMO8BfA+4DPkX0LtIFjPNDZF48PAF8CjgbeCZyY4t+PiOMjYjwwBDgjV+ewiPhoRHwX+L/A30XEh4DWXJkZwI6IOB44HvicpLF04/MmaSjw1vTZz2sGjomIb6Xtz0TEV4CzgD9GxIRcgqwJTiS149X0AZ8QEefs5zF+AHxWUh3wKeBH+yj3HUmrgJlk/yMLuEzSI8C9QAN7X9L4S+DHEdEaEZuApfvZRuufVkTEhoh4A1gFjEnxkyQtl7QaOBk4JlfnVgBJhwFvi4i28bb85/LjwLT0eVwODAfGUezzJkos513LPEZS23az58vEQd0o/1PgG2T/062MiO37KPeViFjUtiPpfKAeOC4iXpe0bh/v5/85B641QGc3T+zKbbcCgyUdBFwNNEXEekmX0v5z83J67ti7zRPZpalftAtKp9PF5y0iXpT0sqR3RsSfci99EPhVZ3VrjXsktW0dcFza/mRXhdPg4i+Aa8guJXTXocDWlEROAkoNRj4ITE3XtEeSXQKzgWMpcKCkz7UF0rjGRzup05Y0npV0CPtIRBHxPLBT0sQUmpp7+RfABZIOSO/5bklvpfuft+8AcyUNSfVPJRsX2VdvvCa5R1LbvglcJ+kSsm5/d9wM/DXZwGJ33QzcIamZ7LLFEyXK3EZ26WI18Af8jW9AiYiQdA7wL5IuIhs3W0c2LtKwjzovSLqW7DOxDniok7eYAVwr6WXgAWBHiv+A7DLZw2lcbhtwNt3/vM0DDgdWS2oF/gxMiYhXuzzpGuIpUqwsyn5TcmhE/GO122LWRtIhEdF2B+BFwMiImF3lZtUM90is29LdXu8i+yZn1pd8QtLFZH/TngbOr25zaot7JGZmVogH283MrBAnEjMzK8SJxMzMCnEiMTOzQpxIzLrQNrFkld57kqQu5x3rbjmzSnAiMevbJtG9CSy7W86sxzmRmHVTmnr8O2lK89WSPpXiJafJlzRG2aJH10paI+mXbVNt7OP4X0zTmj+ibAGoMcDfAX/fNjW5pDPTJIa/k3SvpBH7KNduoSbtma5/pKQHU7lHa226c6sM/47ErAuSXoqIQyR9kuwP9mTgCLIpO/4z2bQbB6dJ/o4Afks2w+w7gBaySQdXSVoILI6IH+7jfTYBYyNil6TD0hQhlwIvRcT/TmUOB15IU478d+C9EfHlEuVuAO5smzwzdw5fBg6KiDlpFueDI2Jnz/+rWS3xL9vNuu8jpKnHgS3KFlU6HribbJr8vwTeoP00+WsjYlXaXsme6dFLeQS4WdLPgZ/vo8wo4NY00eBbgLVlnsNDwPVpEsOf59pmtt98acus+/Y1Xfln2DNN/gRgC3tmrt1revROjv8J4CqyGZlXSipVdh7ZYk/vAz7Pvqf/f3OJgDRZ4VsAIuJBsrU4NgI3SZrWSXvMusWJxKz7HgQ+laYeryf7g7yC7k2T3ylJg4DREXE/8FXgMOAQYCfwtlzRQ8mSAMD0XLxjuXXsWSJgCtA2jfo7UluvBa4jW1vDrBAnErPuu43s8tPvydbX+KzpOtQAAACRSURBVGpE/JlsmvymNE3+Zyg9TX5X6oAfppUAfwdcEREvAHcA52jPOuCXAj+R9Gvg2Vz9juWuBT4qaQXZOE7bIlCTgFWSfke2Bs2V+9FWs3Y82G5mZoW4R2JmZoX4ri2zXibpKuDEDuErI6Kc5YvN+gxf2jIzs0J8acvMzApxIjEzs0KcSMzMrBAnEjMzK+T/Az8CF7svr7/AAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x='loan_status',y='loan_amnt',data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>loan_status</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Charged Off</th>\n",
       "      <td>77673.0</td>\n",
       "      <td>15126.300967</td>\n",
       "      <td>8505.090557</td>\n",
       "      <td>1000.0</td>\n",
       "      <td>8525.0</td>\n",
       "      <td>14000.0</td>\n",
       "      <td>20000.0</td>\n",
       "      <td>40000.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fully Paid</th>\n",
       "      <td>318357.0</td>\n",
       "      <td>13866.878771</td>\n",
       "      <td>8302.319699</td>\n",
       "      <td>500.0</td>\n",
       "      <td>7500.0</td>\n",
       "      <td>12000.0</td>\n",
       "      <td>19225.0</td>\n",
       "      <td>40000.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                count          mean          std     min     25%      50%  \\\n",
       "loan_status                                                                 \n",
       "Charged Off   77673.0  15126.300967  8505.090557  1000.0  8525.0  14000.0   \n",
       "Fully Paid   318357.0  13866.878771  8302.319699   500.0  7500.0  12000.0   \n",
       "\n",
       "                 75%      max  \n",
       "loan_status                    \n",
       "Charged Off  20000.0  40000.0  \n",
       "Fully Paid   19225.0  40000.0  "
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby('loan_status').describe()['loan_amnt']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype=object)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['grade'].sort_values().unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1',\n",
       "       'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2',\n",
       "       'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3',\n",
       "       'G4', 'G5'], dtype=object)"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['sub_grade'].sort_values().unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29c018d6280>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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d9KtWzcAHn30Ua699lo6derE4sWLOe+88ygrK+PCCy/kyiuv3O/+KGBERJLk7gwePJjhw4cza9YsAEpLS1m/fv0Bb7uqqormzaP5J3ny5Mm88MILvPrqqxx++OEsWLCAAQMGUFZWRlZWFtOmTeOuu+7izDPP5IMPPuCFF15IuEpAfSlgRESStHjxYjIzM7nkkkt2l9WsrLxkyRK2b9/OkCFDWL16NYWFhfzpT3/CzLjhhht48skn2bFjB71792bKlCmYGWeccQa9e/fm+eefZ8CAARQXFzNy5EhatWrFN7/5TZ5++mlWr15NdXU1V111FUuWLOGTTz5hzJgxXHzxxbg7l112GYsWLaJLly7UdeH8jTfeyJIlSzj88MMB6NevH7179+bBBx/kvffeY9myZZSXlzNgwAD++te/smHDBgoKCpg0aRLf+ta39vv70hyMiEiSaoKjLn//+9+57bbbeO2113jrrbd4/vnnARg7diwrVqxg9erV7NixY4/DaB999BHPPvssP//5z7nwwgu55557WL58ORkZGbvrTJs2jXbt2rFixQpWrFjB1KlTKS8v32Np/qlTpya86di//vUvPv74Y0444YQ9youKiigrK+Paa6+lqKiIBx98kIkTJzJ//nxOOOEESktLDyhcQAEjItJgevToQW5uLs2aNaOgoIB169YBsZFPz549yc/PZ9GiRZSVle1uc+655wKxoNm2bRu9e/cG4Ec/+tHuOgsWLGDmzJkUFBTQs2dPNm/ezJo1aw5oaX53j3zVZQWMiEiSunXrtnvV5ERatGix+3lGRgZVVVXs3LmTn/70p8yZM4dVq1YxatSoPZbYb9WqFUCdh7dq3ps0aRKlpaWUlpZSXl6+e5HMfYVE27ZtadWqFW+99dYe5a+88gpdu3bda9sDpTkYOWhpsVE52PTp04drrrmGqVOnMmrUKABWrFjBv//97zrb1ITJ0Ucfzfbt25kzZ84XzvICOPLII2nTpg0vvvgivXr12n0SAUD//v25++676dOnD5mZmfzzn/8kJyeH4uJipkyZwrBhw9iwYQOLFy/eY+RTY9y4cfzsZz/j0UcfpWXLlvztb39j2bJlTJky5UC/kr1SwIhIo5XMacUNycyYO3cuV1xxBRMmTCArK2v3acrvvfdewjZHHHEEo0aNIj8/n7y8PE477bQ6tz9t2jRGjRpF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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='grade',data=df,hue='loan_status')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29c018a8f70>"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12,4))\n",
    "sns.countplot(x=df['sub_grade'].sort_values(),palette='coolwarm')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29c010697f0>"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAuAAAAEHCAYAAADvd/OuAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3de5hU1Zno/+8rEFARNVxUIKY5GXOUW1pp0KCCgRl04iUw0ahRR6NidCSjOUd+kzi/Y0wyTtQYjTHqxIQRzdFgokGNmhxnBEWjUUBRJB4jRhxRBPFCxIgCrvNH7SZFW1VdVV1dfeH7eZ56umrtvd9ae+3VxcvqtVdFSglJkiRJ9bFdR1dAkiRJ2paYgEuSJEl1ZAIuSZIk1ZEJuCRJklRHJuCSJElSHfXs6ArU24ABA1JDQ0NHV0OSJEnd3OLFi9emlAa2LN/mEvCGhgYWLVrU0dWQJElSNxcRLxYqdwqKJEmSVEcm4JIkSVIdmYBLkiRJdbTNzQGXJEnqbjZu3MjKlSvZsGFDR1dlm9SnTx+GDh1Kr169ytrfBFySJKmLW7lyJTvttBMNDQ1EREdXZ5uSUuL1119n5cqVDBs2rKxjnIIiSZLUxW3YsIH+/fubfHeAiKB///4V/fXBBFySJKkbMPnuOJW2vQm4JEmSVEcm4JIkSVIdeROmlOeehetLbv/s2L51qokkSR2rb9++rF9f+t/F9nL//ffzkY98hPHjx9dkv87GEXBJkiR1Kvfffz8PP/xwzfbrbEzAJUmSVFRKiZkzZzJy5EhGjRrFLbfcAsD69euZPHky++23H6NGjeKOO+4AYMWKFeyzzz5Mnz6dESNGMGXKFN59992i8X/wgx8wfPhwRo8ezXHHHceKFSv4t3/7N6644goaGxt58MEH+dWvfsX+++/Pvvvuy1//9V+zevXqgvudcsop3HrrrVti9+2b+8v1qlWrmDBhAo2NjYwcOZIHH3ywHVusdU5BkSRJUlG//OUvWbJkCU8++SRr165l7NixTJgwgYEDBzJ37lz69evH2rVrOeCAAzjqqKMAeO655/jZz37Gj3/8Y77whS9w2223ceKJJxaMf/HFF/PCCy/Qu3dv3nrrLXbZZRfOPPNM+vbty3nnnQfAm2++ye9+9zsigp/85CdceumlfO973/vQfrNmzSr4HjfffDOHHnoo//zP/8zmzZv585//3A4tVT4TcEmSJBX10EMPcfzxx9OjRw922203Jk6cyMKFC/nbv/1bzj//fBYsWMB2223Hyy+/zOrVqwEYNmwYjY2NAIwZM4YVK1YUjT969GhOOOEEpk6dytSpUwvus3LlSo499lhWrVrF+++/X/YX3jQbO3Ysp556Khs3bmTq1Klb6tZRnIIiSZKkolJKBctvuukmXnvtNRYvXsySJUvYbbfdtnwZTe/evbfs16NHDzZt2lQ0/t13383ZZ5/N4sWLGTNmTMF9v/KVrzBjxgyWLl3Kj370o6JfetOzZ08++OCDLfV+//33AZgwYQILFixgyJAhnHTSSdx4443lnXw7MQGXJElSURMmTOCWW25h8+bNvPbaayxYsIBx48axbt06Bg0aRK9evZg/fz4vvvhixbE/+OADXnrpJT7zmc9w6aWX8tZbb7F+/Xp22mkn3n777S37rVu3jiFDhgBwww03bClvuV9DQwOLFy8G4I477mDjxo0AvPjiiwwaNIjp06dz2mmn8fjjj1fVFrViAi5JkqSipk2bxujRo/nUpz7FpEmTuPTSS9l999054YQTWLRoEU1NTdx0003svffeFcfevHkzJ554IqNGjWLfffflq1/9KrvssgtHHnkkc+fO3XJz5YUXXsgxxxzDwQcfzIABA7Yc33K/6dOn88ADDzBu3DgeffRRdtxxRyC3WkpjYyP77rsvt912G+ecc07N2qcaUezPCt1VU1NTWrRoUUdXQ52U64BLkrqiZ555hn322aejq7FNK3QNImJxSqmp5b6OgEuSJEl15CookiRJandnn302v/3tb7cqO+ecc/jSl77UQTXqOCbgUjtxOoskSX9x9dVXd3QVOg2noEiSJEl1ZAIuSZIk1ZEJuCRJklRHzgGXJEnqZlq7D6lS5dy31KNHD0aNGrXl9e23305DQ0PBfWfPns2iRYv44Q9/yIUXXkjfvn0577zzyqrLKaecwgMPPMDOO+/Mdtttx9VXX82nP/3povuPHz+ehx9+uGCcI444gqOPPrqs960lE3BJkiS12fbbb8+SJUvq8l7f/e53Ofroo7n33nv58pe/zFNPPVV030LJd0dzCookSZLaRUNDA2vXrgVg0aJFHHLIIUX3ff7559lvv/22vH7uuecYM2ZMyfgTJkxg+fLlrF+/nsmTJ7Pf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\n",
      "text/plain": [
       "<Figure size 864x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12,4))\n",
    "sns.countplot(x=df['sub_grade'].sort_values(),palette='coolwarm',hue=df['loan_status'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29c00f3be20>"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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PaMtSZKkkho1ahSvvvoq06dPL3cpmTJoS5IkqaSmTp1a7hJKwi9DSpIkSRkwaEuSJEkZMGhLkiRJGTBoS5IkSRnwy5CSJEnt1BuXV7fq/vYa37jFPm+//TYXXXQRc+fOpVu3blRVVXHttdfy1ltvcfXVV/Pb3/62VWvaGqeffjrf+MY3PnE3SYCUEldccQV33HEHEUHfvn25/vrrGTRoEAC/+c1vGD9+PHvssQczZsxg9OjRLFiwgDPOOIPvfve721yPQVuSJElFSSkxatQoTjvtNKZMmQLAvHnzWLp06afe99q1a9luu2yi6Q033MCcOXN4/vnn2XHHHXn00Uc55phjWLBgAZWVlUyaNIkbb7yRww8/nLfffps5c+YUvOvk1jJoS5IkqSgzZsyga9eunHvuuRvaampqAJg5cyYrV67k+OOPZ/78+dTW1vKrX/2KiODyyy/noYceYtWqVQwdOpRbbrmFiOCwww5j6NChPPnkkxxzzDEMGzaMM888k+7du/OVr3yFRx55hPnz57Nu3TouueQSZs6cyYcffsjYsWM555xzSClxwQUXMH36dPr3709zN2L86U9/ysyZM9lxxx0BGDFiBEOHDuXOO+/kzTff5IknnmDhwoUcc8wx/O53v2PZsmXU1NRw3XXXccghh2zzfLlGW5IkSUVZH6Cb89xzz3Httdfywgsv8Nprr/Hkk08CcP755zN37lzmz5/PqlWrPrG85N1332XWrFl873vf44wzzuDmm2/mqaeeoqKiYkOfSZMmsfPOOzN37lzmzp3LxIkTWbhwIVOnTuWll16isbGRiRMnMmfOnM1q+utf/8oHH3zA3nvv/Yn2uro6FixYwPjx46mrq+POO+9kwoQJPPjgg+y9997MmzfvU4VsMGhLkiSplQwePJh+/frRpUsXampqWLRoEdB0JnzIkCFUV1czffp0FixYsGGbE088EWgK3O+//z5Dhw4F4Fvf+taGPo8++iiTJ0+mpqaGIUOGsGLFCl5++WVmz57N6NGjqaioYM8992T48OFF15pSIiJa4adunkFbkiRJRRk0aBANDQ3Nft6tW7cNrysqKli7di2rV6/mH//xH7n33ntpbGxkzJgxrF69ekO/7t27AzS77GP9Z9dddx3z5s1j3rx5LFy4kBEjRgBsMSzvtNNOdO/enddee+0T7c8++ywDBw5scdtPy6AtSZKkogwfPpwPP/yQiRMnbmibO3cus2bNanab9aG6V69erFy5knvvvbdgv1133ZWePXvyhz/8AWDDly0BjjzySG666SbWrFkDwJ/+9Cc++OADhg0bxpQpU1i3bh1LlixhxowZBfc9btw4/umf/olVq1YB8F//9V888cQTnzhrngW/DClJktROFXM5vtYUEUydOpWLLrqIK6+8ksrKyg2X93vzzTcLbrPLLrswZswYqqurqaqq4uCDD252/5MmTWLMmDF0796dww47jJ133hmAs846i0WLFnHQQQeRUqJ379488MADjBo1iunTp1NdXc2+++7LoYceWnC/F1xwAX/5y1+orq6moqKCPfbYg2nTprHDDjt8+klpQbR0mr49q6urS/X19Zu1146bXLIaGiacWrKxJElSx/fiiy8yYMCAcpeRmZUrV9KjRw8ArrzySpYsWcIvfvGLMle1UaH5j4iGlFJdof6e0ZYkSVKb8PDDD/OTn/yEtWvX8vnPf57bb7+93CV9KgZtSZIktQknnnjihquQdAR+GVKSJKkd6ajLftu6bZl3g7YkSVI7UVlZyYoVKwzbJZZSYsWKFVRWVm7Vdi4dkSRJaif69evH4sWLWb58eblL6XQqKyvp16/fVm1j0JYkSWonunbtSv/+/ctdhork0hFJkiQpAwZtSZIkKQMGbUmSJCkDBm1JkiQpAwZtSZIkKQMGbUmSJCkDBm1JkiQpAwZtSZIkKQMGbUmSJCkDBm1JkiQpA96CvROrHTe5ZGM1TDi1ZGNJkiS1BZ7RliRJkjJg0JYkSZIyYNCWJEmSMmDQliRJkjKQWdCOiH+LiGURMT+v7bKIeDMi5uUef5/32fcj4pWIeCkijsxrr42Ixtxnv4yIyKpmSZIkqbVkeUb7duCoAu3XpJRqco//AIiIgcBJwKDcNjdGREWu/03A2cA+uUehfUqSJEltSmZBO6U0G/jvIrsfC0xJKX2YUloIvAIMjojPAjullJ5KKSVgMjAym4olSZKk1lOONdrnR8Qfc0tLds219QX+nNdnca6tb+71pu0FRcTZEVEfEfXLly9v7bolSZKkopU6aN8E7A3UAEuAn+XaC627Ti20F5RSujWlVJdSquvdu/enrVWSJEnaZiUN2imlpSmldSmlj4GJwODcR4uBz+V17Qe8lWvvV6BdkiRJatNKGrRza67X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\n",
      "text/plain": [
       "<Figure size 864x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12,4))\n",
    "sns.countplot(x=df[df['sub_grade']>'F']['sub_grade'].sort_values(),hue=df['loan_status'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['loan_repaid'] = pd.get_dummies(df['loan_status'],drop_first=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>loan_status</th>\n",
       "      <th>loan_repaid</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Charged Off</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>396025</th>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>396026</th>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>396027</th>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>396028</th>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>396029</th>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>396030 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        loan_status  loan_repaid\n",
       "0        Fully Paid            1\n",
       "1        Fully Paid            1\n",
       "2        Fully Paid            1\n",
       "3        Fully Paid            1\n",
       "4       Charged Off            0\n",
       "...             ...          ...\n",
       "396025   Fully Paid            1\n",
       "396026   Fully Paid            1\n",
       "396027   Fully Paid            1\n",
       "396028   Fully Paid            1\n",
       "396029   Fully Paid            1\n",
       "\n",
       "[396030 rows x 2 columns]"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['loan_status','loan_repaid']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29c01abc490>"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.corr()['loan_repaid'].sort_values()[:-1].plot(kind='bar')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "loan_amnt               0.000000\n",
       "term                    0.000000\n",
       "int_rate                0.000000\n",
       "installment             0.000000\n",
       "grade                   0.000000\n",
       "sub_grade               0.000000\n",
       "emp_title               5.789208\n",
       "emp_length              4.621115\n",
       "home_ownership          0.000000\n",
       "annual_inc              0.000000\n",
       "verification_status     0.000000\n",
       "issue_d                 0.000000\n",
       "loan_status             0.000000\n",
       "purpose                 0.000000\n",
       "title                   0.443148\n",
       "dti                     0.000000\n",
       "earliest_cr_line        0.000000\n",
       "open_acc                0.000000\n",
       "pub_rec                 0.000000\n",
       "revol_bal               0.000000\n",
       "revol_util              0.069692\n",
       "total_acc               0.000000\n",
       "initial_list_status     0.000000\n",
       "application_type        0.000000\n",
       "mort_acc                9.543469\n",
       "pub_rec_bankruptcies    0.135091\n",
       "address                 0.000000\n",
       "loan_repaid             0.000000\n",
       "dtype: float64"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()/3960.30"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The job title supplied by the Borrower when applying for the loan.*\n"
     ]
    }
   ],
   "source": [
    "feat_info('emp_title')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. \n"
     ]
    }
   ],
   "source": [
    "feat_info('emp_length')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "173105"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['emp_title'].nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Teacher                             4389\n",
       "Manager                             4250\n",
       "Registered Nurse                    1856\n",
       "RN                                  1846\n",
       "Supervisor                          1830\n",
       "                                    ... \n",
       "Fleet and Permit Sales Manager         1\n",
       "G. Edward Solutions at Microsoft       1\n",
       "East River Medical Imaging             1\n",
       "Security Forces Member                 1\n",
       "Modified Polymer Componets             1\n",
       "Name: emp_title, Length: 173105, dtype: int64"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['emp_title'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop('emp_title',axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['1 year',\n",
       " '10+ years',\n",
       " '2 years',\n",
       " '3 years',\n",
       " '4 years',\n",
       " '5 years',\n",
       " '6 years',\n",
       " '7 years',\n",
       " '8 years',\n",
       " '9 years',\n",
       " '< 1 year']"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "list(df['emp_length'].dropna().sort_values().unique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "emp_length_order = [ '< 1 year',\n",
    "                      '1 year',\n",
    "                     '2 years',\n",
    "                     '3 years',\n",
    "                     '4 years',\n",
    "                     '5 years',\n",
    "                     '6 years',\n",
    "                     '7 years',\n",
    "                     '8 years',\n",
    "                     '9 years',\n",
    "                     '10+ years']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29c01eaf6a0>"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12,4))\n",
    "sns.countplot(df['emp_length'].sort_values(),order=emp_length_order)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29c00fe2f40>"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAucAAAEHCAYAAAANq+jXAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3de5xVZb348c9XQFFu3vAGnYafaYqOjTJeokKTjvr7pagdTTxaaojlUdN+5Uk756dUx/LS3eyikahRaihe6liagOadGcUQyaMJKoqIN5ISE/z+/tgLGqYZGGT27OXM5/167des/aznWev57g0z3/3sZz0rMhNJkiRJtbdBrTsgSZIkqcLkXJIkSSoJk3NJkiSpJEzOJUmSpJIwOZckSZJKonetO1AWW265ZdbV1dW6G5IkSermmpubX8zMwW3tMzkv1NXV0dTUVOtuSJIkqZuLiKfa2+e0FkmSJKkkTM4lSZKkkjA5lyRJkkrCOeeSJEnd2JtvvsmCBQtYtmxZrbvS4/Tt25ehQ4fSp0+fDrcxOZckSerGFixYwIABA6irqyMiat2dHiMzeemll1iwYAHDhg3rcDuntUiSJHVjy5YtY4sttjAx72IRwRZbbLHO31iYnEuSJHVzJua18XZed5NzSZIkqSRMziVJkqSS8IJQ/YMRZ15Z9XM0X/TJqp9DkiR1vv79+7N06dKanHvGjBlsuOGGjBw5slPqlZEj55IkSXpHmDFjBvfcc0+n1SujqiXnEfHTiHghIh5pUbZ5RNwWEY8XPzdrse/siHgiIh6LiANblI+IiNnFvu9FMbM+IjaKiGuK8vsjoq5Fm+OKczweEcdVK0ZJkqSeKjM588wz2XXXXamvr+eaa64BYOnSpYwePZo99tiD+vp6brzxRgDmz5/PzjvvzPjx49lll1044IADeP3119s9/ve+9z2GDx/ObrvtxtixY5k/fz4/+tGP+Pa3v01DQwO///3vufnmm9l7773Zfffd+chHPsKiRYvarHf88cczZcqUVcfu378/AAsXLmTUqFE0NDSw66678vvf/76Kr1jHVHNayyTg+0DLORJnAbdn5vkRcVbx/IsRMRwYC+wCbAf8LiJ2zMwVwA+Bk4D7gP8GDgJuAcYBr2TmeyJiLHABcFREbA6cCzQCCTRHxE2Z+UoVY5UkSepRrr/+embNmsXDDz/Miy++yJ577smoUaMYPHgwU6dOZeDAgbz44ovss88+jBkzBoDHH3+cX/ziF1x22WV8/OMf57rrruPYY49t8/jnn38+8+bNY6ONNuLVV19l00035TOf+Qz9+/fnC1/4AgCvvPIK9913HxHBT37yEy688EK++c1v/kO9iRMntnmOn//85xx44IH8x3/8BytWrOCvf/1rFV6pdVO1kfPMvBN4uVXxocAVxfYVwGEtyq/OzDcycx7wBLBXRGwLDMzMezMzqST6h7VxrCnA6GJU/UDgtsx8uUjIb6OS0EuSJKmT3HXXXRx99NH06tWLrbfemn333ZeZM2eSmXzpS19it9124yMf+QjPPvssixYtAmDYsGE0NDQAMGLECObPn9/u8XfbbTeOOeYYfvazn9G7d9vjyQsWLODAAw+kvr6eiy66iDlz5qxTDHvuuSeXX345EyZMYPbs2QwYMGCd2ldDV8853zozFwIUP7cqyocAz7Sot6AoG1Jsty5frU1mLgeWAFus4Vj/ICJOioimiGhavHjxeoQlSZLUs1TGTf/R5MmTWbx4Mc3NzcyaNYutt9561Y14Ntpoo1X1evXqxfLly9s9/q9//WtOOeUUmpubGTFiRJt1TzvtNE499VRmz57Nj3/843Zv+NO7d2/eeuutVf3+29/+BsCoUaO48847GTJkCJ/4xCe48srqL4qxNmW5ILStFdpzDeVvt83qhZmXZmZjZjYOHjy4Qx2VJElSJbG95pprWLFiBYsXL+bOO+9kr732YsmSJWy11Vb06dOH6dOn89RTT63zsd966y2eeeYZPvzhD3PhhRfy6quvsnTpUgYMGMBrr722qt6SJUsYMqQyBnvFFVesKm9dr66ujubmZgBuvPFG3nzzTQCeeuopttpqK8aPH8+4ceN48MEH39Zr0Zm6OjlfVExVofj5QlG+AHhXi3pDgeeK8qFtlK/WJiJ6A4OoTKNp71iSJEnqJIcffji77bYb73vf+9h///258MIL2WabbTjmmGNoamqisbGRyZMns9NOO63zsVesWMGxxx5LfX09u+++O5/73OfYdNNNOeSQQ5g6deqqCz0nTJjAkUceyYc+9CG23HLLVe1b1xs/fjx33HEHe+21F/fffz/9+vUDKqu6NDQ0sPvuu3Pddddx+umnd9rr83ZFe19JdMrBKyuo/Coz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Nm0ac+bMWdXmqKOOAipJ+muvvcbIkSMB+Nd//ddVdW699VauvPJKGhoa2HvvvXnppZd4/PHHufPOOzn66KPp1asX2223Hfvvv3+H+5qZREQnRN0+k3NJkiRVzS677EJzc3O7+zfaaKNV27169WL58uUsW7aMf/u3f2PKlCnMnj2b8ePHs2zZslX1+vXrB9DulJSV+y6++GJmzZrFrFmzmDdvHgcccADAWhPsgQMH0q9fP5588snVyh988EGGDx++xrbry+RckiRJVbP//vvzxhtvcNlll60qmzlzJnfccUe7bVYm4ltuuSVLly5lypQpbdbbbLPNGDBgAPfddx/AqgtOAQ488EB++MMf8uabbwLwP//zP/zlL39h1KhRXH311axYsYKFCxcyffr0No995pln8tnPfpbXX38dgN/97nfcddddq43OV4MXhEqSJPUgHVn6sDNFBFOnTuWMM87g/PPPp2/fvquWUnz22WfbbLPpppsyfvx46uvrqaurY88992z3+BMnTmT8+PH069eP/fbbj0GDBgFw4oknMn/+fPbYYw8yk8GDB3PDDTdw+OGHM23aNOrr69lxxx3Zd9992zzuaaedxiuvvEJ9fT29evVim2224cYbb2TjjTde/xdlDWJNXwf0JI2NjdnU1FTrbpTCiDOvrPo5mi/6ZNXPIUmSYO7cuey888617kbVLF26lP79+wNw/vnns3DhQr773e/WuFd/19brHxHNmdnYVn1HziVJkvSO9etf/5qvf/3rLF++nHe/+91MmjSp1l1aLybnkiRJesc66qijVq3e0h14QagkSVI35zTm2ng7r7vJuSRJUjfWt29fXnrpJRP0LpaZvPTSS/Tt23ed2tVkWktEfA44EUhgNnACsAlwDVAHzAc+npmvFPXPBsYBK4DPZuZvi/IRwCRgY+C/gdMzMyNiI+BKYATwEnBUZs7vmugkSZLKY+jQoSxYsIDFixfXuis9Tt++fRk6dOg6teny5DwihgCfBYZn5usRcS0wFhgO3J6Z50fEWcBZwBcjYnixfxdgO+B3EbFjZq4AfgicBNxHJTk/CLiFSiL/Sma+JyLGAhcA3WcykiRJUgf16dOHYcOG1bob6qBaTWvpDWwcEb2pjJg/BxwKXFHsvwI4rNg+FLg6M9/IzHnAE8BeEbEtMDAz783K9zRXtmqz8lhTgNFR7Xut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\n",
      "text/plain": [
       "<Figure size 864x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12,4))\n",
    "sns.countplot(df['emp_length'].sort_values(),hue=df['loan_status'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>loan_amnt</th>\n",
       "      <th>term</th>\n",
       "      <th>int_rate</th>\n",
       "      <th>installment</th>\n",
       "      <th>grade</th>\n",
       "      <th>sub_grade</th>\n",
       "      <th>emp_length</th>\n",
       "      <th>home_ownership</th>\n",
       "      <th>annual_inc</th>\n",
       "      <th>verification_status</th>\n",
       "      <th>...</th>\n",
       "      <th>pub_rec</th>\n",
       "      <th>revol_bal</th>\n",
       "      <th>revol_util</th>\n",
       "      <th>total_acc</th>\n",
       "      <th>initial_list_status</th>\n",
       "      <th>application_type</th>\n",
       "      <th>mort_acc</th>\n",
       "      <th>pub_rec_bankruptcies</th>\n",
       "      <th>address</th>\n",
       "      <th>loan_repaid</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>loan_status</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Charged Off</th>\n",
       "      <td>77673</td>\n",
       "      <td>77673</td>\n",
       "      <td>77673</td>\n",
       "      <td>77673</td>\n",
       "      <td>77673</td>\n",
       "      <td>77673</td>\n",
       "      <td>72635</td>\n",
       "      <td>77673</td>\n",
       "      <td>77673</td>\n",
       "      <td>77673</td>\n",
       "      <td>...</td>\n",
       "      <td>77673</td>\n",
       "      <td>77673</td>\n",
       "      <td>77610</td>\n",
       "      <td>77673</td>\n",
       "      <td>77673</td>\n",
       "      <td>77673</td>\n",
       "      <td>72123</td>\n",
       "      <td>77586</td>\n",
       "      <td>77673</td>\n",
       "      <td>77673</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fully Paid</th>\n",
       "      <td>318357</td>\n",
       "      <td>318357</td>\n",
       "      <td>318357</td>\n",
       "      <td>318357</td>\n",
       "      <td>318357</td>\n",
       "      <td>318357</td>\n",
       "      <td>305094</td>\n",
       "      <td>318357</td>\n",
       "      <td>318357</td>\n",
       "      <td>318357</td>\n",
       "      <td>...</td>\n",
       "      <td>318357</td>\n",
       "      <td>318357</td>\n",
       "      <td>318144</td>\n",
       "      <td>318357</td>\n",
       "      <td>318357</td>\n",
       "      <td>318357</td>\n",
       "      <td>286112</td>\n",
       "      <td>317909</td>\n",
       "      <td>318357</td>\n",
       "      <td>318357</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>2 rows × 26 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "             loan_amnt    term  int_rate  installment   grade  sub_grade  \\\n",
       "loan_status                                                                \n",
       "Charged Off      77673   77673     77673        77673   77673      77673   \n",
       "Fully Paid      318357  318357    318357       318357  318357     318357   \n",
       "\n",
       "             emp_length  home_ownership  annual_inc  verification_status  ...  \\\n",
       "loan_status                                                               ...   \n",
       "Charged Off       72635           77673       77673                77673  ...   \n",
       "Fully Paid       305094          318357      318357               318357  ...   \n",
       "\n",
       "             pub_rec  revol_bal  revol_util  total_acc  initial_list_status  \\\n",
       "loan_status                                                                   \n",
       "Charged Off    77673      77673       77610      77673                77673   \n",
       "Fully Paid    318357     318357      318144     318357               318357   \n",
       "\n",
       "             application_type  mort_acc  pub_rec_bankruptcies  address  \\\n",
       "loan_status                                                              \n",
       "Charged Off             77673     72123                 77586    77673   \n",
       "Fully Paid             318357    286112                317909   318357   \n",
       "\n",
       "             loan_repaid  \n",
       "loan_status               \n",
       "Charged Off        77673  \n",
       "Fully Paid        318357  \n",
       "\n",
       "[2 rows x 26 columns]"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby('loan_status').count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "emp_co = df[df['loan_status']=='Charged Off'].groupby('emp_length').count()['loan_status']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "emp_fp = df[df['loan_status']=='Fully Paid'].groupby('emp_length').count()['loan_status']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "emp_len = emp_co/emp_fp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "emp_length\n",
       "1 year       0.248649\n",
       "10+ years    0.225770\n",
       "2 years      0.239560\n",
       "3 years      0.242593\n",
       "4 years      0.238213\n",
       "5 years      0.237911\n",
       "6 years      0.233341\n",
       "7 years      0.241887\n",
       "8 years      0.249625\n",
       "9 years      0.250735\n",
       "< 1 year     0.260830\n",
       "Name: loan_status, dtype: float64"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "emp_len"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29c02000280>"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "emp_len.plot(kind='bar')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop('emp_length',axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "loan_amnt                   0\n",
       "term                        0\n",
       "int_rate                    0\n",
       "installment                 0\n",
       "grade                       0\n",
       "sub_grade                   0\n",
       "home_ownership              0\n",
       "annual_inc                  0\n",
       "verification_status         0\n",
       "issue_d                     0\n",
       "loan_status                 0\n",
       "purpose                     0\n",
       "title                    1755\n",
       "dti                         0\n",
       "earliest_cr_line            0\n",
       "open_acc                    0\n",
       "pub_rec                     0\n",
       "revol_bal                   0\n",
       "revol_util                276\n",
       "total_acc                   0\n",
       "initial_list_status         0\n",
       "application_type            0\n",
       "mort_acc                37795\n",
       "pub_rec_bankruptcies      535\n",
       "address                     0\n",
       "loan_repaid                 0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0              vacation\n",
       "1    debt_consolidation\n",
       "2           credit_card\n",
       "3           credit_card\n",
       "4           credit_card\n",
       "5    debt_consolidation\n",
       "6      home_improvement\n",
       "7           credit_card\n",
       "8    debt_consolidation\n",
       "9    debt_consolidation\n",
       "Name: purpose, dtype: object"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['purpose'].head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0                   Vacation\n",
       "1         Debt consolidation\n",
       "2    Credit card refinancing\n",
       "3    Credit card refinancing\n",
       "4      Credit Card Refinance\n",
       "5         Debt consolidation\n",
       "6           Home improvement\n",
       "7       No More Credit Cards\n",
       "8         Debt consolidation\n",
       "9         Debt Consolidation\n",
       "Name: title, dtype: object"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['title'].head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "A category provided by the borrower for the loan request. \n"
     ]
    }
   ],
   "source": [
    "feat_info('purpose')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The loan title provided by the borrower\n"
     ]
    }
   ],
   "source": [
    "feat_info('title')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop('title',axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of mortgage accounts.\n"
     ]
    }
   ],
   "source": [
    "feat_info('mort_acc')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.0     139777\n",
       "1.0      60416\n",
       "2.0      49948\n",
       "3.0      38049\n",
       "4.0      27887\n",
       "5.0      18194\n",
       "6.0      11069\n",
       "7.0       6052\n",
       "8.0       3121\n",
       "9.0       1656\n",
       "10.0       865\n",
       "11.0       479\n",
       "12.0       264\n",
       "13.0       146\n",
       "14.0       107\n",
       "15.0        61\n",
       "16.0        37\n",
       "17.0        22\n",
       "18.0        18\n",
       "19.0        15\n",
       "20.0        13\n",
       "24.0        10\n",
       "22.0         7\n",
       "21.0         4\n",
       "25.0         4\n",
       "27.0         3\n",
       "23.0         2\n",
       "32.0         2\n",
       "26.0         2\n",
       "31.0         2\n",
       "30.0         1\n",
       "28.0         1\n",
       "34.0         1\n",
       "Name: mort_acc, dtype: int64"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['mort_acc'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "int_rate               -0.082583\n",
       "dti                    -0.025439\n",
       "revol_util              0.007514\n",
       "pub_rec                 0.011552\n",
       "pub_rec_bankruptcies    0.027239\n",
       "loan_repaid             0.073111\n",
       "open_acc                0.109205\n",
       "installment             0.193694\n",
       "revol_bal               0.194925\n",
       "loan_amnt               0.222315\n",
       "annual_inc              0.236320\n",
       "total_acc               0.381072\n",
       "mort_acc                1.000000\n",
       "Name: mort_acc, dtype: float64"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.corr()['mort_acc'].sort_values()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "total_acc_avg = df.groupby('total_acc').mean()['mort_acc']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "def fill_mort_acc(total_acc,mort_acc):\n",
    "    if np.isnan(mort_acc):\n",
    "        return total_acc_avg[total_acc]\n",
    "    else:\n",
    "        return mort_acc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'],x['mort_acc']),axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "loan_amnt                 0\n",
       "term                      0\n",
       "int_rate                  0\n",
       "installment               0\n",
       "grade                     0\n",
       "sub_grade                 0\n",
       "home_ownership            0\n",
       "annual_inc                0\n",
       "verification_status       0\n",
       "issue_d                   0\n",
       "loan_status               0\n",
       "purpose                   0\n",
       "dti                       0\n",
       "earliest_cr_line          0\n",
       "open_acc                  0\n",
       "pub_rec                   0\n",
       "revol_bal                 0\n",
       "revol_util              276\n",
       "total_acc                 0\n",
       "initial_list_status       0\n",
       "application_type          0\n",
       "mort_acc                  0\n",
       "pub_rec_bankruptcies    535\n",
       "address                   0\n",
       "loan_repaid               0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "loan_amnt               0\n",
       "term                    0\n",
       "int_rate                0\n",
       "installment             0\n",
       "grade                   0\n",
       "sub_grade               0\n",
       "home_ownership          0\n",
       "annual_inc              0\n",
       "verification_status     0\n",
       "issue_d                 0\n",
       "loan_status             0\n",
       "purpose                 0\n",
       "dti                     0\n",
       "earliest_cr_line        0\n",
       "open_acc                0\n",
       "pub_rec                 0\n",
       "revol_bal               0\n",
       "revol_util              0\n",
       "total_acc               0\n",
       "initial_list_status     0\n",
       "application_type        0\n",
       "mort_acc                0\n",
       "pub_rec_bankruptcies    0\n",
       "address                 0\n",
       "loan_repaid             0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "395219"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status',\n",
       "       'issue_d', 'loan_status', 'purpose', 'earliest_cr_line',\n",
       "       'initial_list_status', 'application_type', 'address'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.select_dtypes(include='object').columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['term'] = df['term'].apply(lambda x:int(x.split()[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "36    301247\n",
       "60     93972\n",
       "Name: term, dtype: int64"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['term'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop('grade',axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [],
   "source": [
    "subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>loan_amnt</th>\n",
       "      <th>term</th>\n",
       "      <th>int_rate</th>\n",
       "      <th>installment</th>\n",
       "      <th>home_ownership</th>\n",
       "      <th>annual_inc</th>\n",
       "      <th>verification_status</th>\n",
       "      <th>issue_d</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>purpose</th>\n",
       "      <th>...</th>\n",
       "      <th>F1</th>\n",
       "      <th>F2</th>\n",
       "      <th>F3</th>\n",
       "      <th>F4</th>\n",
       "      <th>F5</th>\n",
       "      <th>G1</th>\n",
       "      <th>G2</th>\n",
       "      <th>G3</th>\n",
       "      <th>G4</th>\n",
       "      <th>G5</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>10000.0</td>\n",
       "      <td>36</td>\n",
       "      <td>11.44</td>\n",
       "      <td>329.48</td>\n",
       "      <td>RENT</td>\n",
       "      <td>117000.0</td>\n",
       "      <td>Not Verified</td>\n",
       "      <td>Jan-2015</td>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>vacation</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8000.0</td>\n",
       "      <td>36</td>\n",
       "      <td>11.99</td>\n",
       "      <td>265.68</td>\n",
       "      <td>MORTGAGE</td>\n",
       "      <td>65000.0</td>\n",
       "      <td>Not Verified</td>\n",
       "      <td>Jan-2015</td>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>debt_consolidation</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>15600.0</td>\n",
       "      <td>36</td>\n",
       "      <td>10.49</td>\n",
       "      <td>506.97</td>\n",
       "      <td>RENT</td>\n",
       "      <td>43057.0</td>\n",
       "      <td>Source Verified</td>\n",
       "      <td>Jan-2015</td>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>credit_card</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7200.0</td>\n",
       "      <td>36</td>\n",
       "      <td>6.49</td>\n",
       "      <td>220.65</td>\n",
       "      <td>RENT</td>\n",
       "      <td>54000.0</td>\n",
       "      <td>Not Verified</td>\n",
       "      <td>Nov-2014</td>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>credit_card</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>24375.0</td>\n",
       "      <td>60</td>\n",
       "      <td>17.27</td>\n",
       "      <td>609.33</td>\n",
       "      <td>MORTGAGE</td>\n",
       "      <td>55000.0</td>\n",
       "      <td>Verified</td>\n",
       "      <td>Apr-2013</td>\n",
       "      <td>Charged Off</td>\n",
       "      <td>credit_card</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 57 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   loan_amnt  term  int_rate  installment home_ownership  annual_inc  \\\n",
       "0    10000.0    36     11.44       329.48           RENT    117000.0   \n",
       "1     8000.0    36     11.99       265.68       MORTGAGE     65000.0   \n",
       "2    15600.0    36     10.49       506.97           RENT     43057.0   \n",
       "3     7200.0    36      6.49       220.65           RENT     54000.0   \n",
       "4    24375.0    60     17.27       609.33       MORTGAGE     55000.0   \n",
       "\n",
       "  verification_status   issue_d  loan_status             purpose  ...  F1 F2  \\\n",
       "0        Not Verified  Jan-2015   Fully Paid            vacation  ...   0  0   \n",
       "1        Not Verified  Jan-2015   Fully Paid  debt_consolidation  ...   0  0   \n",
       "2     Source Verified  Jan-2015   Fully Paid         credit_card  ...   0  0   \n",
       "3        Not Verified  Nov-2014   Fully Paid         credit_card  ...   0  0   \n",
       "4            Verified  Apr-2013  Charged Off         credit_card  ...   0  0   \n",
       "\n",
       "   F3  F4  F5  G1  G2 G3 G4  G5  \n",
       "0   0   0   0   0   0  0  0   0  \n",
       "1   0   0   0   0   0  0  0   0  \n",
       "2   0   0   0   0   0  0  0   0  \n",
       "3   0   0   0   0   0  0  0   0  \n",
       "4   0   0   0   0   0  0  0   0  \n",
       "\n",
       "[5 rows x 57 columns]"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['loan_amnt', 'term', 'int_rate', 'installment', 'home_ownership',\n",
       "       'annual_inc', 'verification_status', 'issue_d', 'loan_status',\n",
       "       'purpose', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec',\n",
       "       'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',\n",
       "       'application_type', 'mort_acc', 'pub_rec_bankruptcies', 'address',\n",
       "       'loan_repaid', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5',\n",
       "       'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2',\n",
       "       'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4',\n",
       "       'G5'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['home_ownership', 'verification_status', 'issue_d', 'loan_status',\n",
       "       'purpose', 'earliest_cr_line', 'initial_list_status',\n",
       "       'application_type', 'address'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.select_dtypes(include='object').columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [],
   "source": [
    "feat_dummies = pd.get_dummies(df[['verification_status','application_type','initial_list_status','purpose']],drop_first=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop(['verification_status','application_type','initial_list_status','purpose'],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.concat([df,feat_dummies],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>loan_amnt</th>\n",
       "      <th>term</th>\n",
       "      <th>int_rate</th>\n",
       "      <th>installment</th>\n",
       "      <th>home_ownership</th>\n",
       "      <th>annual_inc</th>\n",
       "      <th>issue_d</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>dti</th>\n",
       "      <th>earliest_cr_line</th>\n",
       "      <th>...</th>\n",
       "      <th>purpose_home_improvement</th>\n",
       "      <th>purpose_house</th>\n",
       "      <th>purpose_major_purchase</th>\n",
       "      <th>purpose_medical</th>\n",
       "      <th>purpose_moving</th>\n",
       "      <th>purpose_other</th>\n",
       "      <th>purpose_renewable_energy</th>\n",
       "      <th>purpose_small_business</th>\n",
       "      <th>purpose_vacation</th>\n",
       "      <th>purpose_wedding</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>10000.0</td>\n",
       "      <td>36</td>\n",
       "      <td>11.44</td>\n",
       "      <td>329.48</td>\n",
       "      <td>RENT</td>\n",
       "      <td>117000.0</td>\n",
       "      <td>Jan-2015</td>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>26.24</td>\n",
       "      <td>Jun-1990</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8000.0</td>\n",
       "      <td>36</td>\n",
       "      <td>11.99</td>\n",
       "      <td>265.68</td>\n",
       "      <td>MORTGAGE</td>\n",
       "      <td>65000.0</td>\n",
       "      <td>Jan-2015</td>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>22.05</td>\n",
       "      <td>Jul-2004</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>15600.0</td>\n",
       "      <td>36</td>\n",
       "      <td>10.49</td>\n",
       "      <td>506.97</td>\n",
       "      <td>RENT</td>\n",
       "      <td>43057.0</td>\n",
       "      <td>Jan-2015</td>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>12.79</td>\n",
       "      <td>Aug-2007</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7200.0</td>\n",
       "      <td>36</td>\n",
       "      <td>6.49</td>\n",
       "      <td>220.65</td>\n",
       "      <td>RENT</td>\n",
       "      <td>54000.0</td>\n",
       "      <td>Nov-2014</td>\n",
       "      <td>Fully Paid</td>\n",
       "      <td>2.60</td>\n",
       "      <td>Sep-2006</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>24375.0</td>\n",
       "      <td>60</td>\n",
       "      <td>17.27</td>\n",
       "      <td>609.33</td>\n",
       "      <td>MORTGAGE</td>\n",
       "      <td>55000.0</td>\n",
       "      <td>Apr-2013</td>\n",
       "      <td>Charged Off</td>\n",
       "      <td>33.95</td>\n",
       "      <td>Mar-1999</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 71 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   loan_amnt  term  int_rate  installment home_ownership  annual_inc  \\\n",
       "0    10000.0    36     11.44       329.48           RENT    117000.0   \n",
       "1     8000.0    36     11.99       265.68       MORTGAGE     65000.0   \n",
       "2    15600.0    36     10.49       506.97           RENT     43057.0   \n",
       "3     7200.0    36      6.49       220.65           RENT     54000.0   \n",
       "4    24375.0    60     17.27       609.33       MORTGAGE     55000.0   \n",
       "\n",
       "    issue_d  loan_status    dti earliest_cr_line  ...  \\\n",
       "0  Jan-2015   Fully Paid  26.24         Jun-1990  ...   \n",
       "1  Jan-2015   Fully Paid  22.05         Jul-2004  ...   \n",
       "2  Jan-2015   Fully Paid  12.79         Aug-2007  ...   \n",
       "3  Nov-2014   Fully Paid   2.60         Sep-2006  ...   \n",
       "4  Apr-2013  Charged Off  33.95         Mar-1999  ...   \n",
       "\n",
       "   purpose_home_improvement  purpose_house  purpose_major_purchase  \\\n",
       "0                         0              0                       0   \n",
       "1                         0              0                       0   \n",
       "2                         0              0                       0   \n",
       "3                         0              0                       0   \n",
       "4                         0              0                       0   \n",
       "\n",
       "   purpose_medical  purpose_moving  purpose_other  purpose_renewable_energy  \\\n",
       "0                0               0              0                         0   \n",
       "1                0               0              0                         0   \n",
       "2                0               0              0                         0   \n",
       "3                0               0              0                         0   \n",
       "4                0               0              0                         0   \n",
       "\n",
       "  purpose_small_business  purpose_vacation  purpose_wedding  \n",
       "0                      0                 1                0  \n",
       "1                      0                 0                0  \n",
       "2                      0                 0                0  \n",
       "3                      0                 0                0  \n",
       "4                      0                 0                0  \n",
       "\n",
       "[5 rows x 71 columns]"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MORTGAGE    198022\n",
       "RENT        159395\n",
       "OWN          37660\n",
       "OTHER          110\n",
       "NONE            29\n",
       "ANY              3\n",
       "Name: home_ownership, dtype: int64"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['home_ownership'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'],'OTHER')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MORTGAGE    198022\n",
       "RENT        159395\n",
       "OWN          37660\n",
       "OTHER          142\n",
       "Name: home_ownership, dtype: int64"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['home_ownership'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [],
   "source": [
    "dummies = pd.get_dummies(df['home_ownership'],drop_first=True)\n",
    "df = df.drop('home_ownership',axis=1)\n",
    "df = pd.concat([df,dummies],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['zip_code'] = df['address'].apply(lambda x:x.split(' ')[-1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0         22690\n",
       "1         05113\n",
       "2         05113\n",
       "3         00813\n",
       "4         11650\n",
       "          ...  \n",
       "396025    30723\n",
       "396026    05113\n",
       "396027    70466\n",
       "396028    29597\n",
       "396029    48052\n",
       "Name: zip_code, Length: 395219, dtype: object"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['zip_code']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [],
   "source": [
    "zip_dummies = pd.get_dummies(df['zip_code'],drop_first=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop(['zip_code','address'],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.concat([df,zip_dummies],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop('issue_d',axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda x:int(x[-4:]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop('earliest_cr_line',axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop('loan_status',axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop('loan_repaid',axis=1).values\n",
    "y = df['loan_repaid'].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import MinMaxScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = MinMaxScaler()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = scaler.fit_transform(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(316175, 78)"
      ]
     },
     "execution_count": 96,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(79044, 78)"
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense,Dropout"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "\n",
    "model.add(Dense(78,activation='relu'))\n",
    "model.add(Dropout(0.3))\n",
    "\n",
    "model.add(Dense(39,activation='relu'))\n",
    "model.add(Dropout(0.3))\n",
    "\n",
    "model.add(Dense(19,activation='relu'))\n",
    "model.add(Dropout(0.3))\n",
    "\n",
    "model.add(Dense(1,activation='sigmoid'))\n",
    "\n",
    "model.compile(optimizer='adam',loss='binary_crossentropy')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/25\n",
      "1236/1236 [==============================] - 6s 4ms/step - loss: 0.3644 - val_loss: 0.2649\n",
      "Epoch 2/25\n",
      "1236/1236 [==============================] - 3s 2ms/step - loss: 0.2692 - val_loss: 0.2630\n",
      "Epoch 3/25\n",
      "1236/1236 [==============================] - 4s 3ms/step - loss: 0.2637 - val_loss: 0.2624\n",
      "Epoch 4/25\n",
      "1236/1236 [==============================] - 4s 3ms/step - loss: 0.2628 - val_loss: 0.2620\n",
      "Epoch 5/25\n",
      "1236/1236 [==============================] - 3s 3ms/step - loss: 0.2617 - val_loss: 0.2628\n",
      "Epoch 6/25\n",
      "1236/1236 [==============================] - 4s 3ms/step - loss: 0.2607 - val_loss: 0.2620\n",
      "Epoch 7/25\n",
      "1236/1236 [==============================] - 5s 4ms/step - loss: 0.2591 - val_loss: 0.2625\n",
      "Epoch 8/25\n",
      "1236/1236 [==============================] - 5s 4ms/step - loss: 0.2605 - val_loss: 0.2623\n",
      "Epoch 9/25\n",
      "1236/1236 [==============================] - 3s 3ms/step - loss: 0.2607 - val_loss: 0.2621\n",
      "Epoch 10/25\n",
      "1236/1236 [==============================] - 4s 3ms/step - loss: 0.2595 - val_loss: 0.2618\n",
      "Epoch 11/25\n",
      "1236/1236 [==============================] - 4s 3ms/step - loss: 0.2588 - val_loss: 0.2617\n",
      "Epoch 12/25\n",
      "1236/1236 [==============================] - 3s 2ms/step - loss: 0.2589 - val_loss: 0.2615\n",
      "Epoch 13/25\n",
      "1236/1236 [==============================] - 3s 3ms/step - loss: 0.2591 - val_loss: 0.2616\n",
      "Epoch 14/25\n",
      "1236/1236 [==============================] - 4s 3ms/step - loss: 0.2601 - val_loss: 0.2619\n",
      "Epoch 15/25\n",
      "1236/1236 [==============================] - 3s 3ms/step - loss: 0.2592 - val_loss: 0.2617\n",
      "Epoch 16/25\n",
      "1236/1236 [==============================] - 3s 3ms/step - loss: 0.2589 - val_loss: 0.2618\n",
      "Epoch 17/25\n",
      "1236/1236 [==============================] - 3s 3ms/step - loss: 0.2597 - val_loss: 0.2623\n",
      "Epoch 18/25\n",
      "1236/1236 [==============================] - 5s 4ms/step - loss: 0.2602 - val_loss: 0.2617\n",
      "Epoch 19/25\n",
      "1236/1236 [==============================] - 4s 3ms/step - loss: 0.2590 - val_loss: 0.2614\n",
      "Epoch 20/25\n",
      "1236/1236 [==============================] - 4s 3ms/step - loss: 0.2581 - val_loss: 0.2614\n",
      "Epoch 21/25\n",
      "1236/1236 [==============================] - 3s 2ms/step - loss: 0.2577 - val_loss: 0.2617\n",
      "Epoch 22/25\n",
      "1236/1236 [==============================] - 4s 3ms/step - loss: 0.2594 - val_loss: 0.2614\n",
      "Epoch 23/25\n",
      "1236/1236 [==============================] - 3s 2ms/step - loss: 0.2582 - val_loss: 0.2621\n",
      "Epoch 24/25\n",
      "1236/1236 [==============================] - 3s 3ms/step - loss: 0.2571 - val_loss: 0.2617\n",
      "Epoch 25/25\n",
      "1236/1236 [==============================] - 4s 3ms/step - loss: 0.2561 - val_loss: 0.2611\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<tensorflow.python.keras.callbacks.History at 0x29c16554a00>"
      ]
     },
     "execution_count": 100,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(x=X_train,y=y_train,\n",
    "         validation_data=(X_test,y_test),epochs=25,\n",
    "         batch_size=256,verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import load_model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('ANN_model.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {},
   "outputs": [],
   "source": [
    "losses = pd.DataFrame(model.history.history)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29c177b91f0>"
      ]
     },
     "execution_count": 104,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAX4AAAD4CAYAAADrRI2NAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3deXxd5X3n8c/v7tLVYkm2ZVvyyk5sIKkwkMUJzQJkEpgsTUwIIRRCCQkkzIRCJpOEaZpJGzrpdFoShhCykgAlJKXD1ryatE7KEtvEYGyDMQbb8ibZkqxdd3vmj3MkXcmSfWXJuvI93/frdV9nueec+xxf+fs85znLNeccIiISHKFiF0BERKaXgl9EJGAU/CIiAaPgFxEJGAW/iEjARIpdgLHMnj3bLVmypNjFEBE5Yaxfv/6Ac25OIcvOyOBfsmQJ69atK3YxREROGGa2o9Bl1dUjIhIwCn4RkYBR8IuIBMyM7OMXkeBJp9M0NzfT399f7KLMaIlEgsbGRqLR6DFvQ8EvIjNCc3MzlZWVLFmyBDMrdnFmJOccBw8epLm5maVLlx7zdtTVIyIzQn9/P3V1dQr9IzAz6urqJn1UpOAXkRlDoX90U/FvVDLB75zj//zrK/z71tZiF0VEZEYrmeA3M767Zju/eaml2EURkRNURUVFsYswLUom+AFqkjHae1PFLoaIyIxWUsFfm4zR1qPgF5HJcc5xyy23sHz5clasWMEDDzwAwN69e1m1ahXnnHMOy5cv57e//S3ZbJZPfvKTQ8v+7d/+bZFLf3QldTlnbTLG/k5dAyxyovsf/7yJzXs6p3SbZy6o4qvvf0NByz788MNs2LCB559/ngMHDnDuueeyatUqfvrTn3LRRRfxpS99iWw2S29vLxs2bGD37t28+OKLAHR0dExpuY8HtfhFREb53e9+x+WXX044HKa+vp63v/3trF27lnPPPZfvf//73H777WzcuJHKykqWLVvG9u3bufHGG3niiSeoqqoqdvGPquRa/G09KZxzuixM5ARWaMv8eHHOjTl/1apVrFmzhkcffZQrr7ySW265hU984hM8//zzPPnkk9x55508+OCD3HvvvdNc4okpuRb/QCZHbypb7KKIyAls1apVPPDAA2SzWVpbW1mzZg0rV65kx44dzJ07l0996lNcc801PPfccxw4cIBcLseHPvQhvva1r/Hcc88Vu/hHVVot/vIYAG09KZLxkto1EZlGH/jAB3j66ac5++yzMTO++c1vMm/ePH74wx9yxx13EI1Gqaio4Ec/+hG7d+/m6quvJpfLAfCNb3yjyKU/upJKx9rkcPAvrC0vcmlE5ETT3d0NePcF3XHHHdxxxx0j3r/qqqu46qqrDlvvRGjl5yuprp6aweDXtfwiIuMqqeCvGwz+bgW/iMh4Sir4B1v8untXRGR8JRX8VYkIkZBxUNfyi4iMq6SC38y85/Uo+EVExlVSwQ9eP79a/CIi4yu54K8pV4tfRORISi74ayv0vB4ROf6O9Oz+119/neXLl09jaSam9IK/PKbr+EVEjqCk7twF7+7djt40mWyOSLjk6jWRYHj8Nti3cWq3OW8FXPJX47596623snjxYm644QYAbr/9dsyMNWvW0N7eTjqd5i//8i+57LLLJvSx/f39fPrTn2bdunVEIhG+9a1vceGFF7Jp0yauvvpqUqkUuVyOn//85yxYsICPfOQjNDc3k81m+fKXv8xHP/rRSe32WEoy+AE6+tLMrogXuTQicqJYvXo1n//854eC/8EHH+SJJ57g5ptvpqqqigMHDnD++edz6aWXTujpv3feeScAGzdu5KWXXuI973kPW7du5a677uJzn/scV1xxBalUimw2y2OPPcaCBQt49NFHATh06NDU7yglHPxtPSkFv8iJ6ggt8+PljW98Iy0tLezZs4fW1lZqamqYP38+N998M2vWrCEUCrF7927279/PvHnzCt7u7373O2688UYATj/9dBYvXszWrVu54IIL+PrXv05zczMf/OAHOeWUU1ixYgVf+MIXuPXWW3nf+97H2972tuOyryXXF5If/CIiE/HhD3+Yhx56iAceeIDVq1dz33330drayvr169mwYQP19fX090/sV/7Ge7b/xz72MR555BHKysq46KKL+PWvf82pp57K+vXrWbFiBV/84hf5i7/4i6nYrcMUFPxmdrGZvWxm28zstjHev8zMXjCzDWa2zszeWui6U03BLyLHavXq1dx///089NBDfPjDH+bQoUPMnTuXaDTKb37zG3bs2DHhba5atYr77rsPgK1bt7Jz505OO+00tm/fzrJly7jpppu49NJLeeGFF9izZw/l5eV8/OMf5wtf+MJxe+rnUbt6zCwM3Am8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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "losses.plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\sanyam ahuja\\anaconda3\\lib\\site-packages\\tensorflow\\python\\keras\\engine\\sequential.py:450: UserWarning: `model.predict_classes()` is deprecated and will be removed after 2021-01-01. Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype(\"int32\")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).\n",
      "  warnings.warn('`model.predict_classes()` is deprecated and '\n"
     ]
    }
   ],
   "source": [
    "predictions = model.predict_classes(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report,confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 6746  8912]\n",
      " [   19 63367]]\n"
     ]
    }
   ],
   "source": [
    "print(confusion_matrix(y_test,predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      0.43      0.60     15658\n",
      "           1       0.88      1.00      0.93     63386\n",
      "\n",
      "    accuracy                           0.89     79044\n",
      "   macro avg       0.94      0.72      0.77     79044\n",
      "weighted avg       0.90      0.89      0.87     79044\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test,predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [],
   "source": [
    "import random"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "loan_amnt           25000.00\n",
       "term                   60.00\n",
       "int_rate               18.24\n",
       "installment           638.11\n",
       "annual_inc          61665.00\n",
       "                      ...   \n",
       "48052                   0.00\n",
       "70466                   0.00\n",
       "86630                   0.00\n",
       "93700                   0.00\n",
       "earliest_cr_year     1996.00\n",
       "Name: 305323, Length: 78, dtype: float64"
      ]
     },
     "execution_count": 110,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "random.seed(101)\n",
    "random_ind = random.randint(0,len(df))\n",
    "\n",
    "new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]\n",
    "new_customer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1]])"
      ]
     },
     "execution_count": 111,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.predict_classes(new_customer.values.reshape(1,78))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0"
      ]
     },
     "execution_count": 112,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.iloc[random_ind]['loan_repaid']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
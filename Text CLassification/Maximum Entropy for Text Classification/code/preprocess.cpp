#include <iostream>
#include <vector>
#include <utility>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <algorithm>
#include <map>

using namespace std;

int n = 1;     //ngram


//get phrase ids for sentence
map<string,int> dict;
void read_dict()
{
    ifstream ifs;
    ifs.open("./stanfordSentimentTreebank/dictionary.txt",ios::in);
    if(! ifs.is_open())
    {
        std::cout <<  "fail to open!\n";
        return;
    }

    string line;
    while(getline(ifs,line))
    {
        transform(line.begin(),line.end(),line.begin(),::tolower);
        int idx = line.rfind("|");    //what if "|" in sentence, so call rfind
        dict[line.substr(0,idx)] = stoi(line.substr(idx+1,line.size()-idx-1));
    }
    ifs.close();
}

//get bin-sentiment 
//0 for negative, 1 for positive
//-1 for neutral which will be removed
//sentiment_map: phrase ids - sentiment
map<int,int> sentiment;
void get_sentiment()
{
    ifstream ifs;
    ifs.open("./stanfordSentimentTreebank/sentiment_labels.txt",ios::in);
    if(! ifs.is_open())
    {
        std::cout <<  "fail to open!\n";
        return;
    }

    string line;
    getline(ifs,line);
    while(getline(ifs,line))
    {
        int idx = line.find("|");
        double tmp = stod(line.substr(idx+1,line.size()-idx-1));
        sentiment[stoi(line.substr(0,idx))] = tmp<=0.4?0:(tmp>0.6?1:-1);   // transfer to binary class
    }
    ifs.close();
}

//get the map of dataset split
//'1'train, '2' test, '3' dev
vector<char> splt;
void get_split()
{
    ifstream ifs;
    ifs.open("./stanfordSentimentTreebank/datasetSplit.txt",ios::in);
    if(! ifs.is_open())
    {
        std::cout <<  "fail to open!\n";
        return;
    }

    string line;
    getline(ifs,line);
    while(getline(ifs,line))
    {
        int idx = line.find(",");
        splt.push_back(line[idx+1]);
    }
    ifs.close();


}


//get ngram dictionary
void ngram()
{
    ifstream ifs;
    ofstream ofs_dict;
    ofstream ofs_train;
    ofstream ofs_test;
    ofstream ofs_dev;

    ofs_dict.open("./processed_data_1-gram/words_dict.txt",ios::out);
    ofs_train.open("./processed_data_1-gram/train.txt",ios::out);
    ofs_test.open("./processed_data_1-gram/test.txt",ios::out);
    ofs_dev.open("./processed_data_1-gram/dev.txt",ios::out);
    ifs.open("./stanfordSentimentTreebank/datasetSentences.txt",ios::in);

    ofs_train << "sentiment|sentence" << endl;
    ofs_test << "sentiment|sentence" << endl;
    ofs_dev << "sentiment|sentence" << endl;

    if((! ifs.is_open()) && (! ofs_dict.is_open())
        && (! ofs_train.is_open())&& (! ofs_test.is_open())&& (! ofs_dev.is_open()))
    {
        std::cout <<  "fail to open!\n";
        return ;
    }

    string line;
    vector<string> words;
    set<string> ngram;
    int idx_sentece;
    getline(ifs,line);   //heads

    int cnt = -1;// for sentence ids

    while(getline(ifs,line))
    {
        cnt ++;
        stringstream ssin(line);
        ssin >> idx_sentece;

        string tmp;
        getline(ssin,tmp);
        string sentence = tmp.substr(1,tmp.size()-1);   //remove the front space, get the whole sentence
        int senti = sentiment[dict[sentence]];

        //remove neural 
        if(senti == -1)
            continue;
        transform(sentence.begin(),sentence.end(),sentence.begin(),::tolower);
        //split dataset
        if(splt[cnt] == '1')
            ofs_train << senti << '|' << sentence << endl;
        else if(splt[cnt] == '2')
            ofs_test << senti << '|' << sentence << endl;
        else if(splt[cnt] == '3')
            ofs_dev << senti << '|' << sentence << endl;


        // get the dictionary for ngram
        stringstream sin(sentence);
        while(sin >> tmp)
            words.push_back(tmp);
        // each word
        for(int i=0;i<=words.size();i++)
        {
            tmp = "";

            // n-gram, for n
            for(int k=0;k<n&&i+k<words.size();k++)   
            {
                if(k==0)
                    tmp = words[i];
                else
                    tmp = tmp + ' ' + words[i+k];

                transform(tmp.begin(),tmp.end(),tmp.begin(),::tolower);
                if(ngram.count(tmp) == 0)
                {
                    ngram.insert(tmp);
                    ofs_dict << tmp << "|" << ngram.size() << endl;
                }
            }
        }
        words.clear();

    }
    ifs.close();
    ofs_dict.close();
    ofs_train.close();
    ofs_test.close();
    ofs_dev.close();
    return ;
}



int main()
{
    read_dict();
    get_sentiment();
    get_split();
    ngram();
    return 0;
}   
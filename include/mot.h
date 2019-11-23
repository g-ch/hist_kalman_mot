//
// Created by cc on 2019/11/23.
//

#ifndef HIST_KALMAN_MOT_MOT_H
#define HIST_KALMAN_MOT_MOT_H

#include <target_in_track.h>
#include <map>
#include <string>

using namespace std;

class MOT{
private:
    map<string, TargetInTrack*> objects_map;

public:
    MOT(){
        cout<<"Multiple Object Tracker created!"<<endl;
        objects_map["aa"] = new TargetInTrack();
    }
    ~MOT(){;}
};


#endif //HIST_KALMAN_MOT_MOT_H

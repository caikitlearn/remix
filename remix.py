import cv2
import os

import numpy as np
import pandas as pd

def export_frames(path,start,end,max_saved=2000):
    c=1
    n_saved=0
    vid=cv2.VideoCapture('videos/{path}.mkv'.format(path=path))
    r,frame=vid.read()
    while r:
        if c>=start and c<=end:
            cv2.imwrite('frames/'+str(c)+'.png',frame)
            n_saved+=1
        c+=1
        if n_saved>max_saved:
            break
        r,frame=vid.read()
    vid.release()

def is_win_screen(frame):
    '''
    determines if a frame contains the win screen summary
    '''
    # white line below Free-for-all
    if frame[30,83:233].mean()>240:
    	# black box to the left of Place
        return frame[86:96,58:70].mean()<1
    return False

def array_to_digit(gray_array,pixels):
    '''
    converts a numpy array into a digit from 0 to 7
    '''
    tko=-1
    intensity_max=-1
    for i in range(7,1,-1):
        intensity=gray_array[pixels[i]].mean()
        if intensity>intensity_max:
            intensity_max=intensity
            tko=i
    return tko,intensity_max

def summarize_video(matchup,write=True,verbose=False,to_csv=True):
    export_path='exported_frames/{}/'.format(matchup)
       
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    
    if not os.path.exists('csv'):
        os.makedirs(csv)
    
    # position of brighter pixels that determine digits from 0 to 7
    pixels={d:np.where(cv2.imread('numbers/{}.png'.format(d)).mean(axis=2)>0) for d in range(8)}
    
    win_frames=[]
    p1_tko=[]
    p1_intensity=[]
    p3_tko=[]
    p3_intensity=[]
    
    # loading video, reading in frame 1
    vid=cv2.VideoCapture('videos/{}.mkv'.format(matchup))
    r,frame=vid.read()

    c=1
    last_valid=-1
    last_valid_frame=None
    
    while r:
        # determine the contents of the last valid win screen frame
        if is_win_screen(frame):
            last_valid=c
            last_valid_frame=frame
        # to avoid duplicates, look at only the last frame in a sequence of frames
        elif last_valid==c-1:
            win_frames.append(last_valid)
            
            # TKO numbers for player 1 and player 3
            p1=last_valid_frame[55:(55+10),157:(157+10)]
            p3=last_valid_frame[55:(55+10),212:(212+10)]
            
            tko1,intensity1=array_to_digit(cv2.cvtColor(p1,cv2.COLOR_BGR2GRAY),pixels)
            tko3,intensity3=array_to_digit(cv2.cvtColor(p3,cv2.COLOR_BGR2GRAY),pixels)
            
            p1_tko.append(tko1)
            p3_tko.append(tko3)
            
            p1_intensity.append(intensity1)
            p3_intensity.append(intensity3)
            
            if write:
                cv2.imwrite(export_path+'{}.png'.format(last_valid),
                            last_valid_frame)
        c+=1
        r,frame=vid.read()
    vid.release()
    
    # edge case: if the last frame is a win screen
    if last_valid==c-1:
        win_frames.append(last_valid)

        # TKO numbers for player 1 and player 3
        p1=last_valid_frame[55:(55+10),157:(157+10)]
        p3=last_valid_frame[55:(55+10),212:(212+10)]

        tko1,intensity1=array_to_digit(cv2.cvtColor(p1,cv2.COLOR_BGR2GRAY),pixels)
        tko3,intensity3=array_to_digit(cv2.cvtColor(p3,cv2.COLOR_BGR2GRAY),pixels)

        p1_tko.append(tko1)
        p3_tko.append(tko3)

        p1_intensity.append(intensity1)
        p3_intensity.append(intensity3)

        if write:
            cv2.imwrite(export_path+'{}.png'.format(last_valid),
                        last_valid_frame)
    
    if verbose:
        print('{} frames processed'.format(c-1))
        
    out=pd.DataFrame({'frame':win_frames,
                      'p1_tko':p1_tko,
                      'p3_tko':p3_tko,
                      'p1_intensity':p1_intensity,
                      'p3_intensity':p3_intensity})
    
    out['diff']=out['p3_tko']-out['p1_tko']
    out['win']=1*(out['diff']>0)
    out.loc[out['diff']==0,'win']=np.nan
    out['frame_diff']=out['frame'].diff()
    out['is_timeout']=1*(out[['p1_tko','p3_tko']].max(axis=1)<7)
    
    if to_csv:
        cols=['frame','p1_tko','p3_tko','diff','win','p1_intensity','p3_intensity','frame_diff']
        out[cols].to_csv('csv/{}.csv'.format(matchup),index=False)
    
    return out

def compute_prob(matchups):
    for matchup in matchups:
        print(matchup)
        out=summarize_video(matchup)
        
        trials=out.shape[0]
        ties=(out['diff']==0).sum()
        timeouts=out['is_timeout'].sum()
        n=trials-ties
        
        print('trials:   {} (n={})'.format(trials,n))
        print('ties:     {} ({:.1f}%)'.format(ties,100*ties/trials))
        print('timeouts: {} ({:.1f}%)'.format(timeouts,100*timeouts/trials))
        
        p1_wins=out.loc[out['diff']!=0,'win'].sum()
        print('p1 win:   {:.0f}/{} ({:.1f}%)'.format(p1_wins,
                                                     n,
                                                     100*p1_wins/(n)))
        print('p3 win:   {:.0f}/{} ({:.1f}%)'.format(n-p1_wins,
                                                     n,
                                                     100*(1-p1_wins/n)))

        print('avg. diff: {:.3f}'.format(out['diff'].mean()))
        print('')
    print('done')

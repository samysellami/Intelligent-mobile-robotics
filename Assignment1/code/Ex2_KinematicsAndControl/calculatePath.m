function [ Path, Path_dist ] = calculatePath( map, source, destination )
%function to return the shortest path between source and destination
dim=size(map);
flag=zeros(dim);

prev=struct('row',{-1},'col',{-1});
prev=repmat(prev,dim(1),dim(2));
found=0;
Path_r=struct('row',{},'col',{}); 
Path=struct('row',{},'col',{}); 
queue=struct('row',{},'col',{}); 
flag(source.row, source.col)=1;
queue(1)=source;
Path_dist=[];
radius=25;

while(~isempty(queue))&&(found==0)
    v=queue(1);
    queue(1)=[];
    for i= -1:1
        for j=-1:1
            if ((i==0)&&(j==0))
            else
                w.row=v.row+i;
                w.col=v.col+j;
                if ((w.row==destination.row)&&(w.col==destination.col))
                    found=1;
                    prev(w.row,w.col)=v;
                    disp(w);
                    queue(end+1)=w;
                    i=2;
                    break;
                end
                
                if(~nearObstacle( map, w, radius))%(map(w.row, w.col)==0)
                    if (flag(w.row, w.col)==0) 
                        flag(w.row, w.col)=1;
                        prev(w.row,w.col)=v;
                        queue(end+1)=w;
                    end
                end
            end     
        end
    end           
end

if (found)
    current=destination;
    while (current.row~=-1)
        Path_r(end+1)=current;
        current=prev(current.row, current.col);
    end
    
    Path(1)=Path_r(end);
    Path_dist(1,2)=(-Path(1).row+256)*(5/512);
    Path_dist(1,1)=(Path(1).col-256)*(5/512);

    i=1;
    while(Path(end).row~=destination.row||Path(end).col~=destination.col)
        Path(end+1)= Path_r(end-i);
        i=i+1;
        Path_dist(i,2)=(-Path(end).row+256)*(5/512);
        Path_dist(i,1)=(Path(end).col-256)*(5/512);
    end 

else 
    disp('no destination found!!');
end



 

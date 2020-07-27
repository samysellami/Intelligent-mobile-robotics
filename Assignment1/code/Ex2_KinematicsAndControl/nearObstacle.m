function [obst]=  nearObstacle( map, vertex, radius )
obst= 0;
for i=-radius:radius:radius
    for j=-radius:radius:radius
        if (vertex.row+i<512&& vertex.row+i>0 && vertex.col+j<512&& vertex.col+j>0)
            w.row=vertex.row+i;
            w.col=vertex.col+j;
            if (map(w.row, w.col)==1)
                obst=1;
                return;
            end
        end
    end
end
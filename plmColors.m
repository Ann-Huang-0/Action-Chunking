function map = plmColors(n,c)
% n - number of colors

switch c
    case 'b'
        map = colormap(brewermap(n+1,'Blues'));
        map = map(2:end,:);
        
    case 'g'
        map = colormap(brewermap(n+1,'Greens'));
        map = map(2:end,:);
    case'r'
        map = colormap(brewermap(n+1,'Reds'));
        map = map(2:end,:);
        
    case'k'
        map = colormap(brewermap(n+1,'Greys'));
        map = map(2:end,:);
    
    case 'pastel2'
        map = colormap(brewermap(n, 'Pastel2'));
        
    case 'pastel1'
        map = colormap(brewermap(n, 'Pastel1'));
        
    case 'set2'
        map = colormap(brewermap(n, 'Set2'));
        
end

set(0, 'DefaultAxesColorOrder', map);
%set(gca,'ColorOrder','factory')

%set(0, 'DefaultLineLineWidth', 2) % set line width
end
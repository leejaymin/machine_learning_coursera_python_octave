Update = 0;

data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);

for i=1:97,
    fprintf('%f \n',X(i,:))
end
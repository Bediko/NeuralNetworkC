require 'awesome_print'
require 'benchmark'
def testing(folder,mode,outputs,classes,eventsperclass)
	puts "Testing #{folder} #{mode}"
	test=`./main #{folder}`.split("\n")
	indexmean=test.index("Mittelwert")
	events=[]
	(0..classes-1).each do |i|
		events<<test[indexmean+i+1].split("\t")
		events[i].shift
		events[i].map! { |e| e=Float(e)  }
	end
	(0..classes-1).each do |i|
		file1=IO.readlines("#{folder}/Class#{i}")
		file1.map! do |line|
			line=line.split(" ")
			line.map! do |string|
				string=Float(string)
			end
		end
		(0..outputs-1).each do |j|
			mean=0;
			nEvents=0;
			file1.each do |arr|
				mean+=arr[0]*arr[j+1]
				nEvents+=arr[j+1]
			end
			mean=mean/eventsperclass[i]
			if nEvents!=eventsperclass[i]
				puts "Anzahl der Events stimmen nicht im Histogramm"
				puts "wirklich: #{nEvents}"
				puts "erwartet: #{eventsperclass[i]}"
				puts "Class#{i}"
				puts "Output#{j}"
			end
			error=(mean.round(3)-events[i][j].round(3)).abs
			if error>0.1
				puts "Werte in Histogramm falsch"
				puts "wirklich:#{mean}"
				puts "erwartet:#{events[i][j]}"
				puts "Class#{i}"
				puts "Output#{j}"
			end
		end
	end
	puts "OK"
end

def changeMode(folder,mode,modeline)
	file1=IO.readlines("#{folder}/numbers.txt")
	oldmode=file1[modeline]
	file1[modeline]=mode
	File.open("#{folder}/numbers.txt","w")do |f2|
		file1.each do |string|
			f2.puts string
		end
	end
	return oldmode
end

 $modeline=9
 $oldmode=changeMode("Classification",1,9)
 time=Benchmark.realtime{testing("Classification","sequential",1,2,[3000,3000])}
puts time
 changeMode("Classification",2,9)
 time=Benchmark.realtime{testing("Classification","batch",1,2,[3000,3000])}
puts time
 changeMode("Classification",3,9)
 time=Benchmark.realtime{testing("Classification","mixed",1,2,[3000,3000])}
puts time 
changeMode("Classification",$oldmode,9)

$modeline=10
$oldmode=changeMode("Multiclass",1,10)
time=Benchmark.realtime{testing("Multiclass","sequential",4,4,[1000,1000,1000,1000])}
puts time
changeMode("Multiclass",2,10)
time=Benchmark.realtime{testing("Multiclass","batch",4,4,[1000,1000,1000,1000])}
puts time
changeMode("Multiclass",3,10)
time=Benchmark.realtime{testing("Multiclass","mixed",4,4,[1000,1000,1000,1000])}
puts time
changeMode("Multiclass",$oldmode,10)

$oldmode=changeMode("mittel",1,10)
time=Benchmark.realtime{testing("mittel","sequential",3,3,[4500,1395,112])}
changeMode("mittel",2,10)
puts time
time=Benchmark.realtime{testing("mittel","batch",3,3,[4500,1395,112])}
changeMode("mittel",3,10)
puts time
time=Benchmark.realtime{testing("mittel","mixed",3,3,[4500,1395,112])}
changeMode("mittel",$oldmode,10)
puts time
$oldmode=changeMode("gross",1,10)
time=Benchmark.realtime{testing("gross","sequential",3,3,[94097,1395,112])}
puts time
changeMode("gross",2,10)
time=Benchmark.realtime{testing("gross","batch",3,3,[94097,1395,112])}
puts time
changeMode("gross",3,10)
time=Benchmark.realtime{testing("gross","mixed",3,3,[94097,1395,112])}
puts time
changeMode("gross",$oldmode,10)

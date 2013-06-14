require 'awesome_print'
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
				puts nEvents
				puts eventsperclass[i]
				puts "Class#{i}"
				puts "Output#{j}"
				exit
			end
			error=(mean.round(3)-events[i][j].round(3)).abs
			if error>0.1
				puts "Werte in Histogramm falsch"
				puts nEvents
				puts mean
				puts events[i][j]
				puts "Class#{i}"
				puts "Output#{j}"
				exit(0)
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


 testing("Classification","sequential",1,2,[3000,3000])
 oldmode=changeMode("Classification",2,9)
 testing("Classification","batch",1,2,[3000,3000])
 changeMode("Classification",3,9)
 testing("Classification","mixed",1,2,[3000,3000])
 changeMode("Classification",oldmode,9)

testing("Multiclass","sequential",4,4,[1000,1000,1000,1000])
oldmode=changeMode("Multiclass",2,10)
testing("Multiclass","batch",4,4,[1000,1000,1000,1000])
changeMode("Multiclass",3,10)
testing("Multiclass","mixed",4,4,[1000,1000,1000,1000])
changeMode("Multiclass",oldmode,10)
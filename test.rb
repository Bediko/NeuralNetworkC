require 'awesome_print'
def testing(folder,mode,outputs,classes,eventsperclass)
	puts "Testing #{folder} #{mode}"
	test=`./main #{folder}`.split("\n")
	indexmean=test.index("Mittelwert")
	events=[]
	(0..classes-1).each do |i|
		events<<test[indexmean+i+1].split("\t")
		events[i].shift
		events[i].map! { |e| e=Float(e).round(3)  }
	end
	(0..classes-1).each do |i|
		file1=IO.readlines("Multiclass/Class#{i}")
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
				puts arr[0]
				puts arr[j+1]
				puts mean
			end
			mean=mean/eventsperclass[i]
			if mean.round(3)!=events[i]
			puts "Werte in Histogramm falsch"
			puts i
			puts j
			puts mean
			puts events[i]
			elsif nEvents!=1000
				puts "Anzahl der Events stimmen nicht im Histogramm"
			else
				puts "OK"
			end
		end
	end
end


testing("Classification","sequential",1,2,[1000,1000])
	
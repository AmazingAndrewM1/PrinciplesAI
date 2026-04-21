$command = "python gym_mountain_car_start.py --record --algorithm dqn --episodes 10"
$splitCommand = $command.Split(" ", 2)

$curr = Get-Location
Set-Location Homework2
conda activate temp-env
try{
    $env:PYGAME_DETECT_AVX2 = 1 # This does not do anything even though the documentation says it should...
    $env:KMP_DUPLICATE_LIB_OK = $True
    Start-Process -FilePath $splitCommand[0] -ArgumentList $splitCommand[1] -RedirectStandardOutput ".\out.txt" -NoNewWindow -Wait
    python gym_mountain_car_start.py --algorithm dqn --record
}
catch{
    Write-Host $_
}
finally{
    Set-Location $curr
}
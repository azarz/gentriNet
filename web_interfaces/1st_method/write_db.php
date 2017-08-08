<?php
	$id = $_GET["id"];
	$lat = $_GET["lat"];
	$lon = $_GET["lon"];
	$date1 = $_GET["date1"];
	$date2 = $_GET["date2"];
	$YN = $_GET["YN"];

	$line = array($id, $lat, $lon, $date1, $date2, $YN);

	$handle = fopen("train.csv", "a");
	fputcsv($handle, $line);

	fclose($handle);
?>
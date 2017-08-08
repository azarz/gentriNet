<?php
	$id = $_GET["id"];
	$YN = $_GET["YN"];

	$line = array($id, $YN);

	$handle = fopen("train_review.csv", "a");
	fputcsv($handle, $line);

	fclose($handle);

	echo json_encode(array('newindex' => $newindex))
?>
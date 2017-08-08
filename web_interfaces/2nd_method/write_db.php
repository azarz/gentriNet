<?php
	$id1 = $_GET["id1"];
	$pano1 = $_GET["pano1"];
	$id2 = $_GET["id2"];
	$pano2 = $_GET["pano2"];

	$winner = $_GET["winner"];

	$line = array($id1, $id2, $pano1, $pano2, $winner);

	$handle = fopen("votes.csv", "a");
	fputcsv($handle, $line);

	fclose($handle);

	$images = glob('../voteimages/*.jpg');

	$cnt = count($images)-1;
	$ind1 = rand(0, $cnt);
	$ind2 = rand(0, $cnt);

	echo json_encode(array('id1' => $ind1, 'id2' => $ind2));
?>
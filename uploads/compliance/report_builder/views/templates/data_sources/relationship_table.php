<?php

defined('BASEPATH') or exit('No direct script access allowed');

$aColumns = [
	'id',
	'query_string',
	'left_table',
	'right_table',
];
$sIndexColumn = 'id';
$sTable = db_prefix() . 'rb_data_source_relationships';

$where = [];
$join= [];

$report_template_id = $this->ci->input->post('report_template_id');
if($this->ci->input->post('report_template_id')){
	$where_report_template_id = '';
	$report_template_id = $this->ci->input->post('report_template_id');
	if($report_template_id != '')
	{
		if($where_report_template_id == ''){
			$where_report_template_id .= ' where templates_id = "'.$report_template_id. '"';
		}else{
			$where_report_template_id .= ' or templates_id = "' .$report_template_id.'"';
		}
	}
	if($where_report_template_id != '')
	{
		array_push($where, $where_report_template_id);
	}
}


$result = data_tables_init($aColumns, $sIndexColumn, $sTable, $join, $where, ['id', 'templates_id',	'left_table', 'left_field_1', 'left_field_2', 'right_table', 'right_field_1', 'right_field_2',]);

$output = $result['output'];
$rResult = $result['rResult'];

foreach ($rResult as $aRow) {
	$row = [];

	for ($i = 0; $i < count($aColumns); $i++) {

		if($aColumns[$i] == 'id') {
			$_data = $aRow['id'];

		}elseif ($aColumns[$i] == 'query_string') {
			$code =  $aRow['query_string'];
			$code .= '<div class="row-options">';

			if (has_permission('report_builder', '', 'delete') || is_admin()) {
				$code .= '<a href="' . admin_url('report_builder/delete_relationship/' . $aRow['id'].'/'.$aRow['templates_id']) . '" class="text-danger _delete">' . _l('delete') . '</a>';
			}
			$code .= '</div>';

			$_data = $code;


		}elseif($aColumns[$i] == 'left_table'){
			$_data = $aRow['left_table'];
		}elseif($aColumns[$i] == 'right_table'){
			$_data = $aRow['right_table'];
		}

		$row[] = $_data;
	}

	$output['aaData'][] = $row;
}


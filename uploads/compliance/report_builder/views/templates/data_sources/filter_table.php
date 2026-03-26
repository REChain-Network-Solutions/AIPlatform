<?php

defined('BASEPATH') or exit('No direct script access allowed');

$aColumns = [
	'id',
	'query_filter',
	'group_condition',
	'ask_user',
	'1',
];
$sIndexColumn = 'id';
$sTable = db_prefix() . 'rb_data_source_filters';

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


$result = data_tables_init($aColumns, $sIndexColumn, $sTable, $join, $where, ['id', 'templates_id',	'filter_type', 'filter_value_1', 'filter_value_2', 'field_name', 'table_name', 'filter_type', 'ask_user']);

$output = $result['output'];
$rResult = $result['rResult'];
$rb_filter_type = rb_filter_type();

foreach ($rb_filter_type as $filter_value) {
    $rb_filter_type[$filter_value['name']] = $filter_value;
}

foreach ($rResult as $aRow) {
	$row = [];

	for ($i = 0; $i < count($aColumns); $i++) {

		if($aColumns[$i] == 'id') {
			$_data = $aRow['id'];

		}elseif ($aColumns[$i] == 'query_filter') {
			if($aRow['ask_user'] == 'no'){
				$filter_value_1 = $aRow['filter_value_1'];
				$filter_value_2 = $aRow['filter_value_2'];
			}else{
				$filter_value_1 = ': '._l('rb_ask_user');
				$filter_value_2 = ': '._l('rb_ask_user');
			}

			switch($aRow['filter_type']) {
				case 'equal':

					$code =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;
				break;

				case 'greater_than':
					$code =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'less_than':
					$code =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'greater_than_or_equal':
					$code =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'less_than_or_equal':
					$code =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'between':
					$code =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l('between').' '.$filter_value_1.' '._l('and'). ' '. $filter_value_2;

				break;

				case 'like':
					$code =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'NOT_like':
					$code =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'not_equal':
					$code =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'begin_with':
					$code =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'end_with':
					$code =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'in':
					$code =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' ('.$filter_value_1.')';

				break;

				case 'not_in':
					$code =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' ('.$filter_value_1.')';

				break;
			}


			$_data = $code;


		}elseif($aColumns[$i] == 'group_condition'){
			$_data = $aRow['group_condition'];
		}elseif($aColumns[$i] == 'ask_user'){
			$_data = _l($aRow['ask_user']);

		}elseif($aColumns[$i] == '1'){
			$option = '';
			$option .= '<div class="row">';
			if (has_permission('report_builder', '', 'edit') || is_admin()) {

				$option .= '<a class="btn btn-warning btn-xs mleft5" data-toggle="tooltip" title="" href="javascript:void(0)" onclick="add_filter('.$aRow['templates_id'].', \'update\', '.$aRow['id'].'); return false;"  data-original-title="'._l('edit').'"><i class="fa-regular fa-pen-to-square"></i></a>';
			}

			if (has_permission('report_builder', '', 'delete') || is_admin()) {

				$option .= '<a class="btn btn-danger btn-xs mleft5 _delete" data-toggle="tooltip" title="" href="' . admin_url('report_builder/delete_filter/' . $aRow['id'].'/'.$aRow['templates_id']) . '" data-original-title="'._l('delete').'"><i class="fa fa-remove"></i></a>';
			}
			$option .= '</div>';

			$_data = $option;

		}

		$row[] = $_data;
	}

	$output['aaData'][] = $row;
}


<?php

defined('BASEPATH') or exit('No direct script access allowed');

$aColumns = [
	'table_name',
	'color_hex',
	'id',
];
$sIndexColumn = 'id';
$sTable = db_prefix() . 'rb_field_conditional_formattings';

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


$result = data_tables_init($aColumns, $sIndexColumn, $sTable, $join, $where, ['id', 'templates_id',	'table_name', 'field_name', 'filter_type', 'filter_value_1', 'filter_value_2', 'color_hex',]);

$output = $result['output'];
$rResult = $result['rResult'];

foreach ($rResult as $aRow) {
	$row = [];

	for ($i = 0; $i < count($aColumns); $i++) {

		$_data = $aRow[$aColumns[$i]];

		if($aColumns[$i] == 'color_hex'){
			$_data = '<span class="label" style="background-color: '.$aRow['color_hex'].' ;">&nbsp;</span>'.$aRow['color_hex'];
		}else if($aColumns[$i] == 'filter_type'){
			$_data = _l($aRow['filter_type']);
		}else if($aColumns[$i] == 'id'){
			$option = '';
			$option .= '<div class="row">';
			$option .= '<a href="#" onclick="add_cell_formatting('.$aRow['id'].', '.$aRow['templates_id'].',\'edit\'); return false;"  class="btn btn-icon btn-warning "><i class="fa-regular fa-pen-to-square"></i></a>&nbsp';
			$option .= '<a href="'.admin_url('report_builder/delete_cell_formatting/'. $aRow['id'].'/'.$aRow['templates_id']).'" class="btn btn-icon btn-danger _delete"><i class="fa fa-remove"></i></a>';
			$option .= '</div>';

			$_data = $option; 
		}elseif($aColumns[$i] == 'table_name'){
			//condition formarting
			$condition_formatting = '';
			$filter_value_1 = $aRow['filter_value_1'];
			$filter_value_2 = $aRow['filter_value_2'];
			
			switch($aRow['filter_type']) {
				case 'equal':

					$condition_formatting =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;
				break;

				case 'greater_than':
					$condition_formatting =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'less_than':
					$condition_formatting =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'greater_than_or_equal':
					$condition_formatting =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'less_than_or_equal':
					$condition_formatting =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'between':
					$condition_formatting =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l('between').' '.$filter_value_1.' '._l('and'). ' '. $filter_value_2;

				break;

				case 'like':
					$condition_formatting =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'NOT_like':
					$condition_formatting =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'not_equal':
					$condition_formatting =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'begin_with':
					$condition_formatting =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'end_with':
					$condition_formatting =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' '.$filter_value_1;

				break;

				case 'in':
					$condition_formatting =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' ('.$filter_value_1.')';

				break;

				case 'not_in':
					$condition_formatting =  _l('tbl'.$aRow['table_name'].'_'.$aRow['field_name']).' ('._l('tbl'.$aRow['table_name']).')'.' '._l($aRow['filter_type']).' ('.$filter_value_1.')';

				break;
			}

			$_data = $condition_formatting;

		}

		$row[] = $_data;
	}

	$output['aaData'][] = $row;
}


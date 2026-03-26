<?php

defined('BASEPATH') or exit('No direct script access allowed');

$aColumns = [
	db_prefix() . 'rb_templates.id',
	'report_title',
	'category_id',
	'staff_create',
	'date_create',
	'1',
];
$sIndexColumn = 'id';
$sTable = db_prefix() . 'rb_templates';

$where = [];
$join = [
	'LEFT JOIN ' . db_prefix() . 'rb_categories ON ' . db_prefix() . 'rb_templates.category_id = ' . db_prefix() . 'rb_categories.id',
];

$category_filter = $this->ci->input->post('category_filter');
if(isset($category_filter)){
	$where_category = '';
	foreach ($category_filter as $category_id) {

		if($category_id != '')
		{
			if($where_category == ''){
				$where_category .= ' ('.db_prefix().'rb_templates.category_id in ('.$category_id.')';
			}else{
				$where_category .= ' or '.db_prefix().'rb_templates.category_id in ('.$category_id.')';
			}
		}
	}
	if($where_category != '')
	{
		$where_category .= ')';
		if($where != ''){
			array_push($where, 'AND'. $where_category);
		}else{
			array_push($where, $where_category);
		}
		
	}
}

if(!is_admin()){
	$where[] = 'AND '.db_prefix().'rb_templates.staff_create = '.get_staff_user_id();
}

$result = data_tables_init($aColumns, $sIndexColumn, $sTable, $join, $where, [db_prefix() . 'rb_templates.id', 'name']);

$output = $result['output'];
$rResult = $result['rResult'];

foreach ($rResult as $aRow) {
	$row = [];

	for ($i = 0; $i < count($aColumns); $i++) {

		if($aColumns[$i] == db_prefix() . 'rb_templates.id') {
			$_data = $aRow['id'];

		}elseif ($aColumns[$i] == 'report_title') {
			$code = '<a href="' . admin_url('report_builder/data_source_manage/' . $aRow['id']) . '">' . $aRow['report_title'] . '</a>';

			$_data = $code;


		}elseif($aColumns[$i] == 'category_id'){
			$_data =  $aRow['name'];

		}elseif($aColumns[$i] == 'staff_create'){
			$_data =  get_staff_full_name($aRow['staff_create']);

		}elseif($aColumns[$i] == 'date_create'){
			
			$_data =  _dt($aRow['date_create']);
		}elseif($aColumns[$i] == '1'){
			$option = '';
			if (has_permission('report_builder', '', 'edit') || is_admin()) {

				$option .= '<a class="btn btn-warning btn-xs mleft5" data-toggle="tooltip" title="" href="' . admin_url('report_builder/data_source_manage/' . $aRow['id']) . '" data-original-title="'._l('edit').'"><i class="fa-regular fa-pen-to-square"></i></a>';
			}

			if (has_permission('report_builder', '', 'delete') || is_admin()) {

				$option .= '<a class="btn btn-danger btn-xs mleft5 _delete" data-toggle="tooltip" title="" href="' . admin_url('report_builder/delete_template/' . $aRow['id']) . '" data-original-title="'._l('delete').'"><i class="fa fa-remove"></i></a>';
			}

			$_data = $option;

		}

		$row[] = $_data;
	}

	$output['aaData'][] = $row;
}


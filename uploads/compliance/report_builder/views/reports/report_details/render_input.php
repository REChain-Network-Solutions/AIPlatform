<?php 
if(isset($rb_report_detail)){
	$input_name_id = '#'.$filter_value['id'];
}else{
	$input_name_id = '';
}

if(isset($rb_primary_foreign_key_field[db_prefix().$filter_value['table_name'].'_'.$filter_value['field_name']]) ){
	$select_attrs=[];
	$select_multiple_attrs=[];
	$select_attrs = ['data-toggle' => 'tooltip','data-original-title' => _l('filter_value_title'), 'required' => true, 'data-actions-box' => true];
	$select_multiple_attrs = ['data-toggle' => 'tooltip','data-original-title' => _l('filter_value_title'), 'required' => true, 'multiple' => true, 'data-actions-box' => true];

	//Data source filter
	if(!isset($rb_report_detail)){
		if(new_strlen($filter_value['filter_value_1']) > 0){
			$filter_value['filter_value_1'] = new_explode(',', $filter_value['filter_value_1']);
		}else{
			$filter_value['filter_value_1'] = $filter_value['filter_value_1'];
		}
	}
	
	// operator: = ; IN ; NOT IN : 
	//input type: select single, multiple
	
	$primary_foreign_key = rb_primary_foreign_key_get_data($rb_primary_foreign_key_field[db_prefix().$filter_value['table_name'].'_'.$filter_value['field_name']]);
	switch ($filter_value['filter_type']) {

		case 'equal':
		echo '<div class="col-md-12">'.render_select('filter_value_1'.$input_name_id,$primary_foreign_key,array('value','label'),$filter_value['filter_type'], $filter_value['filter_value_1'], $select_attrs, [], '', '', true).'</div>';

		break;

		case 'in':
		echo '<div class="col-md-12">'.render_select('filter_value_1'.$input_name_id.'[]',$primary_foreign_key,array('value','label'),$filter_value['filter_type'], $filter_value['filter_value_1'], $select_multiple_attrs, [], '', '', false).'</div>';
		break;

		case 'not_in':
		echo '<div class="col-md-12">'.render_select('filter_value_1'.$input_name_id.'[]',$primary_foreign_key,array('value','label'),$filter_value['filter_type'], $filter_value['filter_value_1'], $select_multiple_attrs, [], '', '', false).'</div>';
		break;
		
		default:
			// code...
		break;
	}

}elseif(isset($rb_number_field[db_prefix().$filter_value['table_name'].'_'.$filter_value['field_name']])){

	// operator: = ; > ; < ; >= ; <= ; between ; != 
	//input type: input number
	$input_attrs=[];
	$input_attrs = ['data-toggle' => 'tooltip','data-original-title' => _l('filter_value_title_number'), 'required' => true];

	switch ($filter_value['filter_type']) {

		case 'equal':
		echo '<div class="col-md-12">'.render_input('filter_value_1'.$input_name_id, $filter_value['filter_type'], $filter_value['filter_value_1'],'number',$input_attrs).'</div>';

		break;

		case 'greater_than':
		echo '<div class="col-md-12">'.render_input('filter_value_1'.$input_name_id, $filter_value['filter_type'], $filter_value['filter_value_1'],'number',$input_attrs).'</div>';
		break;

		case 'less_than':
		echo '<div class="col-md-12">'.render_input('filter_value_1'.$input_name_id, $filter_value['filter_type'], $filter_value['filter_value_1'],'number',$input_attrs).'</div>';
		break;

		case 'greater_than_or_equal':
		echo '<div class="col-md-12">'.render_input('filter_value_1'.$input_name_id, $filter_value['filter_type'], $filter_value['filter_value_1'],'number',$input_attrs).'</div>';
		break;

		case 'less_than_or_equal':
		echo '<div class="col-md-12">'.render_input('filter_value_1'.$input_name_id, $filter_value['filter_type'], $filter_value['filter_value_1'],'number',$input_attrs).'</div>';
		break;

		case 'between':
		echo '<div class="col-md-6">'.render_input('filter_value_1'.$input_name_id, $filter_value['filter_type'], $filter_value['filter_value_1'],'number',$input_attrs).'</div>';
		echo '<div class="col-md-6">'.render_input('filter_value_2'.$input_name_id, 'and', $filter_value['filter_value_2'],'number',$input_attrs).'</div>';
		break;

		case 'not_equal':
		echo '<div class="col-md-12">'.render_input('filter_value_1'.$input_name_id, $filter_value['filter_type'], $filter_value['filter_value_1'],'number',$input_attrs).'</div>';
		break;
		
		default:
			// code...
		break;
	}

	
}elseif(isset($rb_date_field[db_prefix().$filter_value['table_name'].'_'.$filter_value['field_name']])){
	// operator: between 
	//input type: input date (two input)
	$input_attrs=[];
	$input_attrs = [ 'required' => true];

	switch ($filter_value['filter_type']) {

		case 'between':
		echo '<div class="col-md-6">'.render_date_input('filter_value_1'.$input_name_id, $filter_value['filter_type'], $filter_value['filter_value_1'],$input_attrs).'</div>';
		echo '<div class="col-md-6">'.render_date_input('filter_value_2'.$input_name_id, 'and', $filter_value['filter_value_2'],$input_attrs).'</div>';

		break;

		default:
			// code...
		break;
	}
	
}elseif(isset($rb_datetime_field[db_prefix().$filter_value['table_name'].'_'.$filter_value['field_name']])){
	// operator: between 
	//input type: input datetime (two input)
	$input_attrs=[];
	$input_attrs = ['data-toggle' => 'tooltip','data-original-title' => _l('filter_value_title'), 'required' => true];
	
	switch ($filter_value['filter_type']) {

		case 'between':
		echo '<div class="col-md-6">'.render_datetime_input('filter_value_1'.$input_name_id, $filter_value['filter_type'], $filter_value['filter_value_1'],$input_attrs).'</div>';
		echo '<div class="col-md-6">'.render_datetime_input('filter_value_2'.$input_name_id, 'and', $filter_value['filter_value_2'],$input_attrs).'</div>';

		break;

		default:
			// code...
		break;
	}
	
}else{
	// operator: = ; like ; not like ; begin with ; end with ; IN ; NOT IN ; 
	//input type: input text
	$input_attrs=[];
	$input_attrs = ['data-toggle' => 'tooltip','data-original-title' => _l('filter_value_title'), 'required' => true];

	$input_attrs_in=[];
	$input_attrs_in = ['data-toggle' => 'tooltip','data-original-title' => _l('filter_value_title_in'), 'required' => true];

	$input_attrs_not_in=[];
	$input_attrs_not_in = ['data-toggle' => 'tooltip','data-original-title' => _l('filter_value_title_not_in'), 'required' => true];

	

	switch ($filter_value['filter_type']) {

		case 'equal':
		echo '<div class="col-md-12">'.render_input('filter_value_1'.$input_name_id, $filter_value['filter_type'], $filter_value['filter_value_1'],'text',$input_attrs).'</div>';

		break;

		case 'like':
		echo '<div class="col-md-12">'.render_input('filter_value_1'.$input_name_id, $filter_value['filter_type'], $filter_value['filter_value_1'],'text',$input_attrs).'</div>';

		break;

		case 'NOT_like':
		echo '<div class="col-md-12">'.render_input('filter_value_1'.$input_name_id, $filter_value['filter_type'], $filter_value['filter_value_1'],'text',$input_attrs).'</div>';

		break;

		case 'begin_with':
		echo '<div class="col-md-12">'.render_input('filter_value_1'.$input_name_id, $filter_value['filter_type'], $filter_value['filter_value_1'],'text',$input_attrs).'</div>';

		break;

		case 'end_with':
		echo '<div class="col-md-12">'.render_input('filter_value_1'.$input_name_id, $filter_value['filter_type'], $filter_value['filter_value_1'],'text',$input_attrs).'</div>';

		break;

		case 'in':
		echo '<div class="col-md-12">'.render_input('filter_value_1'.$input_name_id, $filter_value['filter_type'], $filter_value['filter_value_1'],'text',$input_attrs_in).'</div>';

		break;

		case 'not_in':
		echo '<div class="col-md-12">'.render_input('filter_value_1'.$input_name_id, $filter_value['filter_type'], $filter_value['filter_value_1'],'text',$input_attrs_not_in).'</div>';
		break;
		
		default:
			// code...
		break;
	}


	
}

?>
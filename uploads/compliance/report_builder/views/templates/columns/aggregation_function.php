<?php defined('BASEPATH') or exit('No direct script access allowed'); ?>

<?php echo form_open(admin_url('report_builder/add_aggregation_function/'.$report_template_id), array('id' => 'add_aggregation_function')); ?>

<?php 
	
	$allow_aggregation_function = 'yes';
	$affected_column = '';
	$group_by = '';
	$statistical_function ='';
	$id ='';

	if($aggregation_function){
		$allow_aggregation_function = $aggregation_function->allow_aggregation_function;
		$affected_column 			= $aggregation_function->affected_column;
		$group_by 					= $aggregation_function->group_by;
		$statistical_function 		= $aggregation_function->statistical_function;
		$id 						= $aggregation_function->id;
	}

 ?>

<?php echo form_hidden('id', $id); ?>

<div class="row">
	<div class="col-md-12">
		<label for="contracts_view" class="pt-2"></label>
		<div class="form-group">
			<input data-can-view="" type="checkbox" class="capability" id="allow_aggregation_function" name="allow_aggregation_function" <?php if($allow_aggregation_function == 'yes'){ echo "checked";}else{ echo "";} ?>>
			<label for="allow_aggregation_function" class="pt-2">
				<?php echo _l('rb_allow_aggregation_functions'); ?>               
			</label>
		</div>
	</div>
	
	<div class="col-md-12">

		<div class="form-group">
			<label for="statistical_function" class="control-label"><?php echo _l('rb_statistical_function'); ?>  <a href="javascript:void(0)" class="pull-right display-block input_method"> <i class="fa fa-question-circle i_tooltip" data-toggle="tooltip" title="" data-original-title="<?php echo _l('rb_select_the_function_which_you_want_to_apply'); ?>"></i></a></label>
			<select name="statistical_function" class="form-control selectpicker"  id="statistical_function"  data-width="100%" data-none-selected-text="<?php echo _l('dropdown_non_selected_tex'); ?>" data-live-search="true" > 
				<option value="" ></option>

				<?php foreach (rb_subtotals_data() as $key => $s_function) { ?>

					<?php 
					$selected = '';
					if($s_function['name'] == $statistical_function){
						$selected .= ' selected';
					}
					?>

					<option value="<?php echo new_html_entity_decode($s_function['name']); ?>" <?php echo new_html_entity_decode($selected); ?> ><?php  echo new_html_entity_decode($s_function['label']); ?></option>
				<?php } ?>
			</select>
		</div>

	</div>

	<div class="col-md-12">
		<div class="form-group">
			<label for="affected_column" class="control-label"><?php echo _l('rb_affected_column'); ?>  <a href="javascript:void(0)" class="pull-right display-block input_method"> <i class="fa fa-question-circle i_tooltip" data-toggle="tooltip" title="" data-original-title="<?php echo _l('rb_aggregation_tooltip'); ?>"></i></a></label>
			<select name="affected_column" class="form-control selectpicker"  id="affected_column"  data-width="100%" data-none-selected-text="<?php echo _l('dropdown_non_selected_tex'); ?>" data-live-search="true" > 
				<?php foreach ($templates_selected_column as $key => $selected_column) { ?>

					<?php 
					$selected = '';
					if($selected_column['id'] == $affected_column){
						$selected .= ' selected';
					}
					?>

					<option value="<?php echo new_html_entity_decode($selected_column['id']); ?>" <?php echo new_html_entity_decode($selected); ?> ><?php  echo new_html_entity_decode(_l('tbl'.$selected_column['table_name'].'_'.$selected_column['field_name'])); ?><small> (<?php  echo new_html_entity_decode(_l('tbl'.$selected_column['table_name'])); ?>)</small></option>
				<?php } ?>
			</select>
		</div>
	</div>

	<div class="col-md-12">
		<div class="form-group">
			<label for="group_by" class="control-label"><?php echo _l('rb_group_by'); ?><a href="javascript:void(0)" class="pull-right display-block input_method"> <i class="fa fa-question-circle i_tooltip" data-toggle="tooltip" title="" data-original-title="<?php echo _l('rb_aggregation_group_by_tooltip'); ?>"></i></a></label>
			<select name="group_by" class="form-control selectpicker" id="group_by"  data-width="100%" data-none-selected-text="<?php echo _l('dropdown_non_selected_tex'); ?>" data-live-search="true" > 
				<?php foreach ($templates_selected_column as $key => $selected_column) { ?>

					<?php 
					$selected = '';
					if($selected_column['id'] == $group_by){
						$selected .= ' selected';
					}
					?>

					<option value="<?php echo new_html_entity_decode($selected_column['id']); ?>" <?php echo new_html_entity_decode($selected); ?> ><?php  echo new_html_entity_decode(_l('tbl'.$selected_column['table_name'].'_'.$selected_column['field_name'])); ?><small> (<?php  echo new_html_entity_decode(_l('tbl'.$selected_column['table_name'])); ?>)</small></option>
				<?php } ?>
			</select>
		</div>
	</div>


</div>
<div class="row">
	<div class="col-md-12">
		<div class="modal-footer">
			<button type="submit" class="btn btn-info"><?php echo _l('submit'); ?></button>
		</div>
	</div>
</div>
<?php echo form_close(); ?>
</body>
</html>

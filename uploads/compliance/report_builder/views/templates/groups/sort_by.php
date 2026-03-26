<?php defined('BASEPATH') or exit('No direct script access allowed'); ?>

<?php echo form_open(admin_url('report_builder/add_sort_by_columns/'.$report_template_id), array('id' => 'add_sort_by_columns')); ?>
<div class="row">
	<div class="col-md-12">
		<?php echo form_hidden('templates_id',$report_template_id); ?>
		
		<?php 
		$index=0;
		 ?>

		<?php foreach ($sort_column_by_template_id as $key => $sort_column) { 
			$index++;
		?>

			<div class="row">
				<div class="col-md-6 hide"> 
					<?php echo render_input('id['.$index.']','id', $sort_column['id']); ?>
				</div>

				<div class="col-md-6"> 
					<?php echo render_select('field_name['.$index.']',$fields,array('label','field_name'),'rb_field', $sort_column['field_name'].'-'.$sort_column['table_name']); ?>
				</div>

				<div class="col-md-6">
					<label for="contracts_view" class="pt-2"></label>
					<div class="form-group">
						<input data-can-view="" type="checkbox" class="capability" id="order_by<?php echo '['.$index.']' ?>" name="order_by<?php echo '['.$index.']' ?>" <?php if($sort_column['order_by'] == "DESC"){ echo "checked";}else{ echo "";} ?>>
						<label for="order_by<?php echo '['.$index.']' ?>" class="pt-2">
							<?php echo _l('rb_descending'); ?>               
						</label>
					</div>
				</div>

			</div>
		<?php } ?>

		<?php 
			$new_index = 5 - $index;
		 ?>

		 <?php for ($i = 0; $i < $new_index; $i++) { 
			$index++;

		 ?>
		 	<div class="row">
				<div class="col-md-6 hide"> 
					<?php echo render_input('id['.$index.']','id', 0); ?>
				</div>

				<div class="col-md-6"> 
					<?php echo render_select('field_name['.$index.']',$fields,array('label','field_name'),'rb_field', ''); ?>
				</div>

				<div class="col-md-6">
					<label for="contracts_view" class="pt-2"></label>
					<div class="form-group">
						<input data-can-view="" type="checkbox" class="capability" id="order_by<?php echo '['.$index.']' ?>" name="order_by<?php echo '['.$index.']' ?>" >
						<label for="order_by<?php echo '['.$index.']' ?>" class="pt-2">
							<?php echo _l('rb_descending'); ?>               
						</label>
					</div>
				</div>

			</div>

		 <?php } ?>

		
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

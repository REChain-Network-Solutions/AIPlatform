<div class="modal fade" id="cellFormattingModal">
	<div class="modal-dialog modal-lg">
		<div class="modal-content">
			<div class="modal-header">
				<button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>

				<h4 class="modal-title"><?php echo new_html_entity_decode(_l('add_rb_conditional_formatting')); ?></h4>
			</div>

			<?php 
				$id = '';
				$field_name = '';
				$filter_type = '';
				$filter_value_1 = '';
				$filter_value_2 = '';
				$color_hex = '#d0b7b7';
				$value2_hide = 'hide';
				$value1_col = '12';
			 	

				if(isset($cell_formatting)){
					$id = $cell_formatting->id;
					$field_name = $cell_formatting->field_name.'-'.$cell_formatting->table_name;
					$filter_type = $cell_formatting->filter_type;
					$filter_value_1 = $cell_formatting->filter_value_1;
					$filter_value_2 = $cell_formatting->filter_value_2;
					$color_hex = $cell_formatting->color_hex;


				}

				if(isset($get_filter_type)){
					$filter_type_option = $get_filter_type;
				}else{
					$filter_type_option = rb_filter_type();
				}

			 ?>

			<?php echo form_open(admin_url('report_builder/add_cell_formatting'), array('id' => 'add_cell_formatting')); ?>
			<div class="modal-body">
				<div class="tab-content">
					<div class="row">
						<div class="col-md-12">
							<?php echo form_hidden('templates_id',$template_id); ?>
							<?php echo form_hidden('id',$id); ?>
							
							<div class="row">
								<div class="col-md-6"> 
									<?php echo render_select('field_name',$fields,array('label','field_name'),'rb_field', $field_name); ?>
								</div>
								<div class="col-md-6"> 
									<?php echo render_select('filter_type', $filter_type_option,array('name','label'),'filter_type', $filter_type); ?>
								</div>
							</div>

							<div class="row">

								<div class="filter_input ">
									<?php 
										if(isset($cell_formatting)){

											$rb_primary_foreign_key_field = rb_primary_foreign_key_field();
											$rb_number_field = rb_number_field();
											$rb_date_field = rb_date_field();
											$rb_datetime_field = rb_datetime_field();

											$data = [];
											$data['rb_primary_foreign_key_field'] = $rb_primary_foreign_key_field;
											$data['rb_number_field'] = $rb_number_field;
											$data['rb_date_field'] = $rb_date_field;
											$data['rb_datetime_field'] = $rb_datetime_field;
											$filter_value_1 = $cell_formatting->filter_value_1;


											$data['filter_value'] = [
												'table_name' => $cell_formatting->table_name,
												'field_name' => $cell_formatting->field_name,
												'filter_type' => $filter_type,
												'filter_value_1' => $filter_value_1,
												'filter_value_2' => $cell_formatting->filter_value_2,
											];


											echo new_html_entity_decode($this->load->view('report_builder/reports/report_details/render_input', $data, true));

										}
									 ?>
								</div>
							</div>

							<div class="row">
								<div class="col-md-12">
									<?php echo render_color_picker('color_hex', _l('rb_color'), $color_hex); ?>
								</div>
							</div>
							

						</div>
					</div>
				</div>
			</div>
			<div class="modal-footer">
				<button type="button" class="btn btn-default close_btn" data-dismiss="modal"><?php echo _l('close'); ?></button>
				<button type="submit" class="btn btn-info"><?php echo _l('submit'); ?></button>
			</div>

		</div>

		<?php echo form_close(); ?>
	</div>
</div>
</div>
<?php require('modules/report_builder/assets/js/templates/cells/cell_modal_js.php'); ?>

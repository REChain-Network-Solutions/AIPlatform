<div class="modal fade" id="filterModal">
	<div class="modal-dialog ">
		<div class="modal-content">
			<div class="modal-header">
				<button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>

				<h4 class="modal-title"><?php echo new_html_entity_decode(_l('rb_prameters')); ?></h4>
			</div>
			<?php echo form_open(admin_url('report_builder/report_detail/'.$template_id), array('id' => 'report_filter_modal')); ?>

			<?php echo form_hidden('filter_value',true); ?>

			<div class="modal-body">
				<div class="tab-content">
					<div class="row">
						<div class="col-md-12">
							<?php 
							$name_date_input = rb_field_name_date_input();	
							$rb_primary_foreign_key_field = rb_primary_foreign_key_field();
							$rb_number_field = rb_number_field();
							$rb_date_field = rb_date_field();
							$rb_datetime_field = rb_datetime_field();

							$data = [];
							$data['rb_primary_foreign_key_field'] = $rb_primary_foreign_key_field;
							$data['rb_number_field'] = $rb_number_field;
							$data['rb_date_field'] = $rb_date_field;
							$data['rb_datetime_field'] = $rb_datetime_field;
							$data['rb_report_detail'] = true;

							$input_attrs=[];
							$input_attrs = ['data-toggle' => 'tooltip','data-original-title' => _l('filter_value_title'), 'required' => true];

							 ?>
							<?php foreach ($filter_data as $filter_value) { 
								if($filter_value['ask_user'] == 'yes'){
									$filter_value['filter_value_1'] = ''; 
									$filter_value['filter_value_2'] = ''; 
									$data['filter_value'] = $filter_value;
									$filter_html = '';
									$filter_html .= '<div class="col-md-12"><h4>'._l('tbl'.$filter_value['table_name'].'_'.$filter_value['field_name']).'</h4></div>';
							?>
							<?php  
							$filter_html .= $this->load->view('report_builder/reports/report_details/render_input', $data, true);

							

							if($filter_html != ''){
								echo '<div class="row">'.$filter_html.'</div>';
							}

							 ?>
							    
							<?php } } ?>

						</div>
					</div>
				</div>
			</div>
			<div class="modal-footer">
				<button type="submit" class="btn btn-info"><?php echo _l('rb_run'); ?></button>
			</div>

		</div>

		<?php echo form_close(); ?>
	</div>
</div>
</div>
<?php require('modules/report_builder/assets/js/reports/report_filter_modal_js.php'); ?>
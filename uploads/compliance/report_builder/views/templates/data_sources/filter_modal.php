<div class="modal fade" id="filterModal">
	<div class="modal-dialog modal-lg">
		<div class="modal-content">
			<div class="modal-header">
				<button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>

				<h4 class="modal-title"><?php echo new_html_entity_decode(_l('add_filter')); ?></h4>
			</div>

			<?php
			 $id = '';
			 $filter_input_hide = '';
			 if(isset($datasource_filter_id)){
			 	$id = $datasource_filter_id;
			 }

			 $table_name = isset($datasource_filter) ? $datasource_filter->table_name : '' ;
			 $field_name = isset($datasource_filter) ? $datasource_filter->field_name : '' ;
			 $filter_type = isset($datasource_filter) ? $datasource_filter->filter_type : '' ;
			 $group_condition = isset($datasource_filter) ? $datasource_filter->group_condition : 'AND' ;

			 if(isset($datasource_filter) && $datasource_filter->ask_user == 'yes'){
			 	$filter_input_hide = ' hide';
			 }

			 $ask_user = '';
			 if(isset($datasource_filter) && $datasource_filter->ask_user == 'yes'){
			 	$ask_user = 'checked';
			 }

			  isset($datasource_filter) ? $datasource_filter->filter_type : '' ;

			 if(isset($datasource_filter)){

			 	$field_name_option = $field_name_option;
			 }else{
			 	$field_name_option = [];
			 }

			 if(isset($get_filter_type)){
			 	$filter_type_option = $get_filter_type;
			 }else{
			 	$filter_type_option = rb_filter_type();
			 }


			 ?>
			<?php echo form_open(admin_url('report_builder/add_filter/'.$id), array('id' => 'add_filter')); ?>
			<div class="modal-body">
				<div class="tab-content">
					<div class="row">
						<div class="col-md-12">
							<?php echo form_hidden('templates_id',$template_id); ?>
							
							<?php 
							$select_attrs=[];
							$select_attrs = ['data-toggle' => 'tooltip','data-original-title' => _l('filter_value_title'), 'required' => true];
							 ?>
							<div class="row">
								<div class="col-md-4"> 
									<?php echo render_select('table_name',$tables,array('name','label'),'table_name', $table_name, $select_attrs); ?>
								</div>
								<div class="col-md-4"> 
									<?php echo render_select('field_name', $field_name_option,array('id','description'),'field_name', $field_name, $select_attrs); ?>
								</div>
								<div class="col-md-4"> 
									<?php echo render_select('filter_type', $filter_type_option,array('name','label'),'filter_type', $filter_type, $select_attrs, [], '', '', false); ?>
								</div>
							</div>
							<div class="row">

								<div class="filter_input <?php echo new_html_entity_decode($filter_input_hide); ?>">
									<?php 
										if($filter_input_hide == '' && isset($datasource_filter)){

											$rb_primary_foreign_key_field = rb_primary_foreign_key_field();
											$rb_number_field = rb_number_field();
											$rb_date_field = rb_date_field();
											$rb_datetime_field = rb_datetime_field();

											$data = [];
											$data['rb_primary_foreign_key_field'] = $rb_primary_foreign_key_field;
											$data['rb_number_field'] = $rb_number_field;
											$data['rb_date_field'] = $rb_date_field;
											$data['rb_datetime_field'] = $rb_datetime_field;
												$filter_value_1 = $datasource_filter->filter_value_1;


											$data['filter_value'] = [
												'table_name' => $table_name,
												'field_name' => $field_name,
												'filter_type' => $filter_type,
												'filter_value_1' => $filter_value_1,
												'filter_value_2' => $datasource_filter->filter_value_2,
											];

											echo new_html_entity_decode($this->load->view('report_builder/reports/report_details/render_input', $data, true));

										}
									 ?>
								</div>
							</div>

							<div class="row">
								<div class="col-md-4"></div>
								<div class="col-md-4">
									<div class="form-group">
										<label>   </label>
										<div class="checkbox checkbox-primary">
											<input  type="checkbox" id="ask_user" name="ask_user" value="yes" <?php echo new_html_entity_decode($ask_user); ?> >
											<label for="ask_user"><?php echo _l('rb_ask_user'); ?>
											 <a href="#" class="pull-right display-block input_method"> <i class="fa fa-question-circle i_tooltip" data-toggle="tooltip" title="" data-original-title="<?php echo _l('rb_ask_user_tooltip'); ?>"></i></a>
										</label>
									</div>
								</div>
								</div>

								<div class="col-md-4">
									<?php 
									$group_condition_data=[];
									$group_condition_data[] = [
										'name' => 'AND',
										'label' => _l('rb_and'),
									];
									$group_condition_data[] = [
										'name' => 'OR',
										'label' => _l('rb_or'),
									];
									
									 ?>

									<div class="form-group">
										<label for="group_condition" class="control-label"><?php echo _l('rb_group_condition'); ?>  <a href="javascript:void(0)" class="pull-right display-block input_method"> <i class="fa fa-question-circle i_tooltip" data-toggle="tooltip" title="" data-original-title="<?php echo _l('rb_group_condition_tooltip'); ?>"></i></a></label>
										<select name="group_condition" class="form-control selectpicker" id="group_condition"  data-width="100%" data-none-selected-text="<?php echo _l('dropdown_non_selected_tex'); ?>" data-live-search="true" > 
											<?php foreach ($group_condition_data as $key => $condition_data) { ?>
												<?php 
													$selected = '';
													if($condition_data['name'] == $group_condition){
														$selected = ' selected';
													}
												 ?>
												<option value="<?php echo new_html_entity_decode($condition_data['name']); ?>" <?php echo new_html_entity_decode($selected); ?>><?php  echo new_html_entity_decode($condition_data['label']); ?></option>
											<?php } ?>
										</select>
									</div>

								</div>
							</div>
							<div class="row hide">
								
								<div class="col-md-12"> 
									<?php echo render_textarea('query_filter','query_filter'); ?>
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
<?php require('modules/report_builder/assets/js/templates/data_source/filters_modal_js.php'); ?>
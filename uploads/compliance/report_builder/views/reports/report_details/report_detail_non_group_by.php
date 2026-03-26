<table id="dtBasicExample"  class="table dt-table border table-striped">
	<thead>
		<?php $header_key = []; ?>
		<?php foreach ($get_selected_column as $report_key => $report_header) { ?>
			<?php 
				$header_key[] = db_prefix().$report_header['table_name'].'.'.$report_header['field_name'];
			 ?>
			<th><?php echo new_html_entity_decode($report_header['label_name']); ?></th>
		<?php } ?>
		</thead>

		<tbody>
			<?php foreach ($report_result as $key => $report_value) { ?>

				<?php if($key > 0){ ?>

					<tr>
						<?php foreach ($report_value as $report_key =>$report_value) { ?>

							<td style="<?php echo rb_cell_formatting_color($report_key, $cell_formattings, $report_value) ?>"><?php echo new_html_entity_decode(rb_report_detail_format_value($report_value, $report_key)); ?></td>

						<?php } ?>
					</tr>

				<?php }else{ ?>
					<?php 
					$first_row='';
					?>
					<?php foreach ($report_value as $report_key => $report_value) { ?>

						<?php $first_row .= '<td style="'. rb_cell_formatting_color($report_key, $cell_formattings, $report_value) .'">'. rb_report_detail_format_value($report_value, $report_key).'</td>'; ?>

					<?php } ?>
					<?php if(new_strlen($first_row) > 0){ ?>
						<tr>
							<?php echo new_html_entity_decode($first_row); ?>
						</tr>
					<?php } ?>


				<?php } ?>

			<?php } ?>



		</tbody>
	</table> 
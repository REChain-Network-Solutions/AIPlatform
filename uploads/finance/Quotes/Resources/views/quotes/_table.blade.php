<table border="1" cellpadding="6" cellspacing="0" width="100%">
  <thead><tr><th align="left">Description</th><th align="right">Qty</th><th align="right">Unit</th><th align="right">Line</th></tr></thead>
  <tbody>
  @foreach($quote->items as $it)
    <tr>
      <td>{{ $it->description }}</td>
      <td align="right">{{ number_format($it->qty, 2) }}</td>
      <td align="right">{{ number_format($it->unit_price, 2) }}</td>
      <td align="right">{{ number_format($it->line_total, 2) }}</td>
    </tr>
  @endforeach
  </tbody>
  <tfoot>
    <tr><td colspan="3" align="right"><strong>Subtotal</strong></td><td align="right">{{ number_format($quote->subtotal, 2) }}</td></tr>
    <tr><td colspan="3" align="right"><strong>Tax</strong></td><td align="right">{{ number_format($quote->tax_total, 2) }}</td></tr>
    <tr><td colspan="3" align="right"><strong>Grand Total</strong></td><td align="right">{{ number_format($quote->grand_total, 2) }}</td></tr>
  </tfoot>
</table>

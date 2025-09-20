# [YooKassa API SDK](../home.md)

# Interface: ReceiptItemInterface
### Namespace: [\YooKassa\Model\Receipt](../namespaces/yookassa-model-receipt.md)
---
**Summary:**

Interface ReceiptItemInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getAdditionalPaymentSubjectProps()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getAdditionalPaymentSubjectProps) |  | Возвращает дополнительный реквизит предмета расчета. |
| public | [getAgentType()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getAgentType) |  | Возвращает тип посредника, реализующего товар или услугу. |
| public | [getAmount()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getAmount) |  | Возвращает общую стоимость покупаемого товара в копейках/центах. |
| public | [getCountryOfOriginCode()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getCountryOfOriginCode) |  | Возвращает код страны происхождения товара по общероссийскому классификатору стран мира. |
| public | [getCustomsDeclarationNumber()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getCustomsDeclarationNumber) |  | Возвращает номер таможенной декларации. |
| public | [getDescription()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getDescription) |  | Возвращает наименование товара. |
| public | [getExcise()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getExcise) |  | Возвращает сумму акциза товара с учетом копеек. |
| public | [getMarkCodeInfo()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getMarkCodeInfo) |  | Возвращает код товара. |
| public | [getMarkMode()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getMarkMode) |  | Возвращает режим обработки кода маркировки. |
| public | [getMarkQuantity()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getMarkQuantity) |  | Возвращает дробное количество маркированного товара. |
| public | [getMeasure()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getMeasure) |  | Возвращает меру количества предмета расчета. |
| public | [getPaymentMode()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getPaymentMode) |  | Возвращает признак способа расчета. |
| public | [getPaymentSubject()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getPaymentSubject) |  | Возвращает признак предмета расчета. |
| public | [getPaymentSubjectIndustryDetails()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getPaymentSubjectIndustryDetails) |  | Возвращает отраслевой реквизит чека. |
| public | [getPrice()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getPrice) |  | Возвращает цену товара. |
| public | [getProductCode()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getProductCode) |  | Возвращает код товара — уникальный номер, который присваивается экземпляру товара при маркировке. |
| public | [getQuantity()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getQuantity) |  | Возвращает количество товара. |
| public | [getSupplier()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getSupplier) |  | Возвращает информацию о поставщике товара или услуги. |
| public | [getVatCode()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_getVatCode) |  | Возвращает ставку НДС |
| public | [isShipping()](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md#method_isShipping) |  | Проверяет, является ли текущий элемент чека доставкой. |

---
### Details
* File: [lib/Model/Receipt/ReceiptItemInterface.php](../../lib/Model/Receipt/ReceiptItemInterface.php)
* Package: \YooKassa\Model
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Interface |
| author |  | cms@yoomoney.ru |
| property |  | Наименование товара (тег в 54 ФЗ — 1030) |
| property |  | Количество (тег в 54 ФЗ — 1023) |
| property |  | Мера количества предмета расчета (тег в 54 ФЗ — 2108) |
| property |  | Дробное количество маркированного товара (тег в 54 ФЗ — 1291) |
| property |  | Дробное количество маркированного товара (тег в 54 ФЗ — 1291) |
| property |  | Суммарная стоимость покупаемого товара в копейках/центах |
| property |  | Цена товара (тег в 54 ФЗ — 1079) |
| property |  | Ставка НДС, число 1-10 (тег в 54 ФЗ — 1199) |
| property |  | Ставка НДС, число 1-10 (тег в 54 ФЗ — 1199) |
| property |  | Признак предмета расчета (тег в 54 ФЗ — 1212) |
| property |  | Признак предмета расчета (тег в 54 ФЗ — 1212) |
| property |  | Признак способа расчета (тег в 54 ФЗ — 1214) |
| property |  | Признак способа расчета (тег в 54 ФЗ — 1214) |
| property |  | Код страны происхождения товара (тег в 54 ФЗ — 1230) |
| property |  | Код страны происхождения товара (тег в 54 ФЗ — 1230) |
| property |  | Номер таможенной декларации (от 1 до 32 символов). Тег в 54 ФЗ — 1231 |
| property |  | Номер таможенной декларации (от 1 до 32 символов). Тег в 54 ФЗ — 1231 |
| property |  | Сумма акциза товара с учетом копеек (тег в 54 ФЗ — 1229) |
| property |  | Код товара — уникальный номер, который присваивается экземпляру товара при маркировке (тег в 54 ФЗ — 1162) |
| property |  | Код товара — уникальный номер, который присваивается экземпляру товара при маркировке (тег в 54 ФЗ — 1162) |
| property |  | Код товара (тег в 54 ФЗ — 1163) |
| property |  | Код товара (тег в 54 ФЗ — 1163) |
| property |  | Режим обработки кода маркировки (тег в 54 ФЗ — 2102) |
| property |  | Режим обработки кода маркировки (тег в 54 ФЗ — 2102) |
| property |  | Отраслевой реквизит предмета расчета (тег в 54 ФЗ — 1260) |
| property |  | Отраслевой реквизит предмета расчета (тег в 54 ФЗ — 1260) |
| property |  | Дополнительный реквизит предмета расчета (тег в 54 ФЗ — 1191) |
| property |  | Дополнительный реквизит предмета расчета (тег в 54 ФЗ — 1191) |
| property |  | Информация о поставщике товара или услуги (тег в 54 ФЗ — 1224) |
| property |  | Тип посредника, реализующего товар или услугу |
| property |  | Тип посредника, реализующего товар или услугу |

---
## Methods
<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает наименование товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** string|null - Наименование товара


<a name="method_getQuantity" class="anchor"></a>
#### public getQuantity() : float|null

```php
public getQuantity() : float|null
```

**Summary**

Возвращает количество товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** float|null - Количество купленного товара


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : int

```php
public getAmount() : int
```

**Summary**

Возвращает общую стоимость покупаемого товара в копейках/центах.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** int - Сумма стоимости покупаемого товара


<a name="method_getPrice" class="anchor"></a>
#### public getPrice() : \YooKassa\Model\AmountInterface

```php
public getPrice() : \YooKassa\Model\AmountInterface
```

**Summary**

Возвращает цену товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** \YooKassa\Model\AmountInterface - Цена товара


<a name="method_getVatCode" class="anchor"></a>
#### public getVatCode() : null|int

```php
public getVatCode() : null|int
```

**Summary**

Возвращает ставку НДС

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** null|int - Ставка НДС, число 1-10, или null, если ставка не задана


<a name="method_getPaymentSubject" class="anchor"></a>
#### public getPaymentSubject() : null|string

```php
public getPaymentSubject() : null|string
```

**Summary**

Возвращает признак предмета расчета.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** null|string - Признак предмета расчета


<a name="method_getPaymentMode" class="anchor"></a>
#### public getPaymentMode() : null|string

```php
public getPaymentMode() : null|string
```

**Summary**

Возвращает признак способа расчета.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** null|string - Признак способа расчета


<a name="method_getProductCode" class="anchor"></a>
#### public getProductCode() : null|string

```php
public getProductCode() : null|string
```

**Summary**

Возвращает код товара — уникальный номер, который присваивается экземпляру товара при маркировке.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** null|string - Код товара


<a name="method_getMarkCodeInfo" class="anchor"></a>
#### public getMarkCodeInfo() : \YooKassa\Model\Receipt\MarkCodeInfo|null

```php
public getMarkCodeInfo() : \YooKassa\Model\Receipt\MarkCodeInfo|null
```

**Summary**

Возвращает код товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** \YooKassa\Model\Receipt\MarkCodeInfo|null - Код товара


<a name="method_getMeasure" class="anchor"></a>
#### public getMeasure() : string|null

```php
public getMeasure() : string|null
```

**Summary**

Возвращает меру количества предмета расчета.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** string|null - Мера количества предмета расчета


<a name="method_getMarkMode" class="anchor"></a>
#### public getMarkMode() : string|null

```php
public getMarkMode() : string|null
```

**Summary**

Возвращает режим обработки кода маркировки.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** string|null - Режим обработки кода маркировки


<a name="method_getMarkQuantity" class="anchor"></a>
#### public getMarkQuantity() : \YooKassa\Model\Receipt\MarkQuantity|null

```php
public getMarkQuantity() : \YooKassa\Model\Receipt\MarkQuantity|null
```

**Summary**

Возвращает дробное количество маркированного товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** \YooKassa\Model\Receipt\MarkQuantity|null - Дробное количество маркированного товара


<a name="method_getPaymentSubjectIndustryDetails" class="anchor"></a>
#### public getPaymentSubjectIndustryDetails() : \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface

```php
public getPaymentSubjectIndustryDetails() : \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает отраслевой реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface - Отраслевой реквизит чека


<a name="method_getAdditionalPaymentSubjectProps" class="anchor"></a>
#### public getAdditionalPaymentSubjectProps() : string|null

```php
public getAdditionalPaymentSubjectProps() : string|null
```

**Summary**

Возвращает дополнительный реквизит предмета расчета.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** string|null - Дополнительный реквизит предмета расчета


<a name="method_getCountryOfOriginCode" class="anchor"></a>
#### public getCountryOfOriginCode() : null|string

```php
public getCountryOfOriginCode() : null|string
```

**Summary**

Возвращает код страны происхождения товара по общероссийскому классификатору стран мира.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** null|string - Код страны происхождения товара


<a name="method_getCustomsDeclarationNumber" class="anchor"></a>
#### public getCustomsDeclarationNumber() : null|string

```php
public getCustomsDeclarationNumber() : null|string
```

**Summary**

Возвращает номер таможенной декларации.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** null|string - Номер таможенной декларации (от 1 до 32 символов)


<a name="method_getExcise" class="anchor"></a>
#### public getExcise() : null|float

```php
public getExcise() : null|float
```

**Summary**

Возвращает сумму акциза товара с учетом копеек.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** null|float - Сумма акциза товара с учетом копеек


<a name="method_getSupplier" class="anchor"></a>
#### public getSupplier() : \YooKassa\Model\Receipt\SupplierInterface|null

```php
public getSupplier() : \YooKassa\Model\Receipt\SupplierInterface|null
```

**Summary**

Возвращает информацию о поставщике товара или услуги.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** \YooKassa\Model\Receipt\SupplierInterface|null - Информация о поставщике товара или услуги


<a name="method_getAgentType" class="anchor"></a>
#### public getAgentType() : string|null

```php
public getAgentType() : string|null
```

**Summary**

Возвращает тип посредника, реализующего товар или услугу.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** string|null - Тип посредника, реализующего товар или услугу


<a name="method_isShipping" class="anchor"></a>
#### public isShipping() : bool

```php
public isShipping() : bool
```

**Summary**

Проверяет, является ли текущий элемент чека доставкой.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptItemInterface](../classes/YooKassa-Model-Receipt-ReceiptItemInterface.md)

**Returns:** bool - True если доставка, false если обычный товар




---

### Top Namespaces

* [\YooKassa](../namespaces/yookassa.md)

---

### Reports
* [Errors - 0](../reports/errors.md)
* [Markers - 0](../reports/markers.md)
* [Deprecated - 40](../reports/deprecated.md)

---

This document was automatically generated from source code comments on 2025-09-04 using [phpDocumentor](http://www.phpdoc.org/)

&copy; 2025 YooMoney
# [YooKassa API SDK](../home.md)

# Interface: PaymentInterface
### Namespace: [\YooKassa\Model\Payment](../namespaces/yookassa-model-payment.md)
---
**Summary:**

Interface PaymentInterface.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
* [public MIN_LENGTH_ID](../classes/YooKassa-Model-Payment-PaymentInterface.md#constant_MIN_LENGTH_ID)
* [public MAX_LENGTH_ID](../classes/YooKassa-Model-Payment-PaymentInterface.md#constant_MAX_LENGTH_ID)
* [public MAX_LENGTH_DESCRIPTION](../classes/YooKassa-Model-Payment-PaymentInterface.md#constant_MAX_LENGTH_DESCRIPTION)
* [public MAX_LENGTH_MERCHANT_CUSTOMER_ID](../classes/YooKassa-Model-Payment-PaymentInterface.md#constant_MAX_LENGTH_MERCHANT_CUSTOMER_ID)

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getAmount()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getAmount) |  | Возвращает сумму. |
| public | [getAuthorizationDetails()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getAuthorizationDetails) |  | Возвращает данные об авторизации платежа. |
| public | [getCancellationDetails()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getCancellationDetails) |  | Возвращает комментарий к статусу canceled: кто отменил платеж и по какой причине. |
| public | [getCapturedAt()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getCapturedAt) |  | Возвращает время подтверждения платежа магазином или null, если время не задано. |
| public | [getConfirmation()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getConfirmation) |  | Возвращает способ подтверждения платежа. |
| public | [getCreatedAt()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getCreatedAt) |  | Возвращает время создания заказа. |
| public | [getDeal()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getDeal) |  | Возвращает сделку, в рамках которой нужно провести платеж. |
| public | [getExpiresAt()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getExpiresAt) |  | Возвращает время до которого можно бесплатно отменить или подтвердить платеж, или null, если оно не задано. |
| public | [getId()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getId) |  | Возвращает идентификатор платежа. |
| public | [getIncomeAmount()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getIncomeAmount) |  | Возвращает сумму перечисляемая магазину за вычетом комиссий платежной системы.(только для успешных платежей). |
| public | [getInvoiceDetails()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getInvoiceDetails) |  | Возвращает invoice_details. |
| public | [getMetadata()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getMetadata) |  | Возвращает метаданные платежа установленные мерчантом |
| public | [getPaid()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getPaid) |  | Проверяет, был ли уже оплачен заказ. |
| public | [getPaymentMethod()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getPaymentMethod) |  | Возвращает используемый способ проведения платежа. |
| public | [getReceiptRegistration()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getReceiptRegistration) |  | Возвращает состояние регистрации фискального чека. |
| public | [getRecipient()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getRecipient) |  | Возвращает получателя платежа. |
| public | [getRefundable()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getRefundable) |  | Возможность провести возврат по API. |
| public | [getRefundedAmount()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getRefundedAmount) |  | Возвращает сумму возвращенных средств. |
| public | [getStatus()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getStatus) |  | Возвращает состояние платежа. |
| public | [getTransfers()](../classes/YooKassa-Model-Payment-PaymentInterface.md#method_getTransfers) |  | Возвращает данные о распределении платежа между магазинами. |

---
### Details
* File: [lib/Model/Payment/PaymentInterface.php](../../lib/Model/Payment/PaymentInterface.php)
* Package: \YooKassa\Model
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |
| property |  | Идентификатор платежа |
| property |  | Текущее состояние платежа |
| property |  | Получатель платежа |
| property |  | Сумма заказа |
| property |  | Описание транзакции |
| property |  | Способ проведения платежа |
| property |  | Способ проведения платежа |
| property |  | Время создания заказа |
| property |  | Время создания заказа |
| property |  | Время подтверждения платежа магазином |
| property |  | Время подтверждения платежа магазином |
| property |  | Время, до которого можно бесплатно отменить или подтвердить платеж |
| property |  | Время, до которого можно бесплатно отменить или подтвердить платеж |
| property |  | Способ подтверждения платежа |
| property |  | Сумма возвращенных средств платежа |
| property |  | Сумма возвращенных средств платежа |
| property |  | Признак оплаты заказа |
| property |  | Возможность провести возврат по API |
| property |  | Состояние регистрации фискального чека |
| property |  | Состояние регистрации фискального чека |
| property |  | Метаданные платежа указанные мерчантом |
| property |  | Признак тестовой операции |
| property |  | Комментарий к отмене платежа |
| property |  | Комментарий к отмене платежа |
| property |  | Данные об авторизации платежа |
| property |  | Данные об авторизации платежа |
| property |  | Данные о распределении платежа между магазинами |
| property |  | Сумма платежа, которую получит магазин |
| property |  | Сумма платежа, которую получит магазин |
| property |  | Данные о сделке, в составе которой проходит платеж |
| property |  | Идентификатор покупателя в вашей системе, например электронная почта или номер телефона |
| property |  | Идентификатор покупателя в вашей системе, например электронная почта или номер телефона |
| property |  | Данные о выставленном счете, в рамках которого проведен платеж |
| property |  | Данные о выставленном счете, в рамках которого проведен платеж |

---
## Constants
<a name="constant_MIN_LENGTH_ID" class="anchor"></a>
###### MIN_LENGTH_ID
```php
MIN_LENGTH_ID = 36 : int
```


<a name="constant_MAX_LENGTH_ID" class="anchor"></a>
###### MAX_LENGTH_ID
```php
MAX_LENGTH_ID = 36 : int
```


<a name="constant_MAX_LENGTH_DESCRIPTION" class="anchor"></a>
###### MAX_LENGTH_DESCRIPTION
Максимальная длина строки описания платежа

```php
MAX_LENGTH_DESCRIPTION = 128
```


<a name="constant_MAX_LENGTH_MERCHANT_CUSTOMER_ID" class="anchor"></a>
###### MAX_LENGTH_MERCHANT_CUSTOMER_ID
Максимальная длина строки идентификатора покупателя в вашей системе

```php
MAX_LENGTH_MERCHANT_CUSTOMER_ID = 200
```



---
## Methods
<a name="method_getId" class="anchor"></a>
#### public getId() : string|null

```php
public getId() : string|null
```

**Summary**

Возвращает идентификатор платежа.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** string|null - Идентификатор платежа


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : string|null

```php
public getStatus() : string|null
```

**Summary**

Возвращает состояние платежа.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** string|null - Текущее состояние платежа


<a name="method_getRecipient" class="anchor"></a>
#### public getRecipient() : null|\YooKassa\Model\Payment\RecipientInterface

```php
public getRecipient() : null|\YooKassa\Model\Payment\RecipientInterface
```

**Summary**

Возвращает получателя платежа.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** null|\YooKassa\Model\Payment\RecipientInterface - Получатель платежа или null, если получатель не задан


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает сумму.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма платежа


<a name="method_getPaymentMethod" class="anchor"></a>
#### public getPaymentMethod() : \YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod|null

```php
public getPaymentMethod() : \YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod|null
```

**Summary**

Возвращает используемый способ проведения платежа.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** \YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod|null - Способ проведения платежа


<a name="method_getCreatedAt" class="anchor"></a>
#### public getCreatedAt() : \DateTime|null

```php
public getCreatedAt() : \DateTime|null
```

**Summary**

Возвращает время создания заказа.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** \DateTime|null - Время создания заказа


<a name="method_getCapturedAt" class="anchor"></a>
#### public getCapturedAt() : null|\DateTime

```php
public getCapturedAt() : null|\DateTime
```

**Summary**

Возвращает время подтверждения платежа магазином или null, если время не задано.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** null|\DateTime - Время подтверждения платежа магазином


<a name="method_getConfirmation" class="anchor"></a>
#### public getConfirmation() : \YooKassa\Model\Payment\Confirmation\AbstractConfirmation|null

```php
public getConfirmation() : \YooKassa\Model\Payment\Confirmation\AbstractConfirmation|null
```

**Summary**

Возвращает способ подтверждения платежа.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** \YooKassa\Model\Payment\Confirmation\AbstractConfirmation|null - Способ подтверждения платежа


<a name="method_getRefundedAmount" class="anchor"></a>
#### public getRefundedAmount() : \YooKassa\Model\AmountInterface|null

```php
public getRefundedAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает сумму возвращенных средств.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма возвращенных средств платежа


<a name="method_getPaid" class="anchor"></a>
#### public getPaid() : bool

```php
public getPaid() : bool
```

**Summary**

Проверяет, был ли уже оплачен заказ.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** bool - Признак оплаты заказа, true если заказ оплачен, false если нет


<a name="method_getRefundable" class="anchor"></a>
#### public getRefundable() : bool

```php
public getRefundable() : bool
```

**Summary**

Возможность провести возврат по API.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** bool - Возможность провести возврат по API


<a name="method_getReceiptRegistration" class="anchor"></a>
#### public getReceiptRegistration() : string|null

```php
public getReceiptRegistration() : string|null
```

**Summary**

Возвращает состояние регистрации фискального чека.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** string|null - Состояние регистрации фискального чека


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает метаданные платежа установленные мерчантом

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** \YooKassa\Model\Metadata|null - Метаданные платежа указанные мерчантом


<a name="method_getExpiresAt" class="anchor"></a>
#### public getExpiresAt() : null|\DateTime

```php
public getExpiresAt() : null|\DateTime
```

**Summary**

Возвращает время до которого можно бесплатно отменить или подтвердить платеж, или null, если оно не задано.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** null|\DateTime - Время, до которого можно бесплатно отменить или подтвердить платеж


<a name="method_getCancellationDetails" class="anchor"></a>
#### public getCancellationDetails() : null|\YooKassa\Model\CancellationDetailsInterface

```php
public getCancellationDetails() : null|\YooKassa\Model\CancellationDetailsInterface
```

**Summary**

Возвращает комментарий к статусу canceled: кто отменил платеж и по какой причине.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** null|\YooKassa\Model\CancellationDetailsInterface - Комментарий к статусу canceled


<a name="method_getAuthorizationDetails" class="anchor"></a>
#### public getAuthorizationDetails() : null|\YooKassa\Model\Payment\AuthorizationDetailsInterface

```php
public getAuthorizationDetails() : null|\YooKassa\Model\Payment\AuthorizationDetailsInterface
```

**Summary**

Возвращает данные об авторизации платежа.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** null|\YooKassa\Model\Payment\AuthorizationDetailsInterface - Данные об авторизации платежа


<a name="method_getTransfers" class="anchor"></a>
#### public getTransfers() : \YooKassa\Model\Payment\TransferInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getTransfers() : \YooKassa\Model\Payment\TransferInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает данные о распределении платежа между магазинами.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** \YooKassa\Model\Payment\TransferInterface[]|\YooKassa\Common\ListObjectInterface - 


<a name="method_getIncomeAmount" class="anchor"></a>
#### public getIncomeAmount() : ?\YooKassa\Model\AmountInterface

```php
public getIncomeAmount() : ?\YooKassa\Model\AmountInterface
```

**Summary**

Возвращает сумму перечисляемая магазину за вычетом комиссий платежной системы.(только для успешных платежей).

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** ?\YooKassa\Model\AmountInterface - 


<a name="method_getDeal" class="anchor"></a>
#### public getDeal() : \YooKassa\Model\Deal\PaymentDealInfo|null

```php
public getDeal() : \YooKassa\Model\Deal\PaymentDealInfo|null
```

**Summary**

Возвращает сделку, в рамках которой нужно провести платеж.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** \YooKassa\Model\Deal\PaymentDealInfo|null - Сделка, в рамках которой нужно провести платеж


<a name="method_getInvoiceDetails" class="anchor"></a>
#### public getInvoiceDetails() : \YooKassa\Model\Payment\PaymentInvoiceDetails|null

```php
public getInvoiceDetails() : \YooKassa\Model\Payment\PaymentInvoiceDetails|null
```

**Summary**

Возвращает invoice_details.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentInterface](../classes/YooKassa-Model-Payment-PaymentInterface.md)

**Returns:** \YooKassa\Model\Payment\PaymentInvoiceDetails|null - 




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
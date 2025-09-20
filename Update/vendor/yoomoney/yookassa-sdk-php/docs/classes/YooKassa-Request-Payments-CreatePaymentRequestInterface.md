# [YooKassa API SDK](../home.md)

# Interface: CreatePaymentRequestInterface
### Namespace: [\YooKassa\Request\Payments](../namespaces/yookassa-request-payments.md)
---
**Summary:**

Interface CreatePaymentRequestInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getAirline()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getAirline) |  | Возвращает данные длинной записи. |
| public | [getAmount()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getAmount) |  | Возвращает сумму заказа. |
| public | [getCapture()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getCapture) |  | Возвращает флаг автоматического принятия поступившей оплаты. |
| public | [getClientIp()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getClientIp) |  | Возвращает IPv4 или IPv6-адрес покупателя. |
| public | [getConfirmation()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getConfirmation) |  | Возвращает способ подтверждения платежа. |
| public | [getDeal()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getDeal) |  | Возвращает данные о сделке, в составе которой проходит платеж. |
| public | [getDescription()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getDescription) |  | Возвращает описание транзакции. |
| public | [getMerchantCustomerId()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getMerchantCustomerId) |  | Возвращает идентификатор покупателя в вашей системе. |
| public | [getMetadata()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getMetadata) |  | Возвращает данные оплаты установленные мерчантом |
| public | [getPaymentMethodData()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getPaymentMethodData) |  | Возвращает данные для создания метода оплаты. |
| public | [getPaymentMethodId()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getPaymentMethodId) |  | Устанавливает идентификатор записи платёжных данных покупателя. |
| public | [getPaymentOrder()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getPaymentOrder) |  | Возвращает платежное поручение. |
| public | [getPaymentToken()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getPaymentToken) |  | Возвращает одноразовый токен для проведения оплаты. |
| public | [getReceipt()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getReceipt) |  | Возвращает чек, если он есть. |
| public | [getReceiver()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getReceiver) |  | Возвращает реквизиты получателя оплаты. |
| public | [getRecipient()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getRecipient) |  | Возвращает объект получателя платежа. |
| public | [getSavePaymentMethod()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getSavePaymentMethod) |  | Возвращает флаг сохранения платёжных данных. |
| public | [getTransfers()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_getTransfers) |  | Возвращает данные о распределении денег — сколько и в какой магазин нужно перевести. |
| public | [hasAirline()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_hasAirline) |  | Проверяет, были ли установлены данные длинной записи. |
| public | [hasCapture()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_hasCapture) |  | Проверяет, был ли установлен флаг автоматического приняти поступившей оплаты. |
| public | [hasClientIp()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_hasClientIp) |  | Проверяет, был ли установлен IPv4 или IPv6-адрес покупателя. |
| public | [hasConfirmation()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_hasConfirmation) |  | Проверяет, был ли установлен способ подтверждения платежа. |
| public | [hasDeal()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_hasDeal) |  | Проверяет, были ли установлены данные о сделке. |
| public | [hasDescription()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_hasDescription) |  | Проверяет наличие описания транзакции в создаваемом платеже. |
| public | [hasMerchantCustomerId()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_hasMerchantCustomerId) |  | Проверяет, был ли установлен идентификатор покупателя в вашей системе. |
| public | [hasMetadata()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_hasMetadata) |  | Проверяет, были ли установлены метаданные заказа. |
| public | [hasPaymentMethodData()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_hasPaymentMethodData) |  | Проверяет установлен ли объект с методом оплаты. |
| public | [hasPaymentMethodId()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_hasPaymentMethodId) |  | Проверяет наличие идентификатора записи о платёжных данных покупателя. |
| public | [hasPaymentOrder()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_hasPaymentOrder) |  | Проверяет, было ли установлено платежное поручение. |
| public | [hasPaymentToken()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_hasPaymentToken) |  | Проверяет наличие одноразового токена для проведения оплаты. |
| public | [hasReceipt()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_hasReceipt) |  | Проверяет наличие чека в создаваемом платеже. |
| public | [hasReceiver()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_hasReceiver) |  | Проверяет, были ли установлены реквизиты получателя оплаты. |
| public | [hasRecipient()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_hasRecipient) |  | Проверяет наличие получателя платежа в запросе. |
| public | [hasSavePaymentMethod()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_hasSavePaymentMethod) |  | Проверяет, был ли установлен флаг сохранения платёжных данных. |
| public | [hasTransfers()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_hasTransfers) |  | Проверяет наличие данных о распределении денег. |
| public | [setAirline()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_setAirline) |  | Устанавливает данные авиабилетов. |
| public | [setCapture()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_setCapture) |  | Устанавливает флаг автоматического принятия поступившей оплаты. |
| public | [setClientIp()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_setClientIp) |  | Устанавливает IP адрес покупателя. |
| public | [setConfirmation()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_setConfirmation) |  | Устанавливает способ подтверждения платежа. |
| public | [setDeal()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_setDeal) |  | Устанавливает данные о сделке, в составе которой проходит платеж. |
| public | [setDescription()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_setDescription) |  | Устанавливает описание транзакции. |
| public | [setMerchantCustomerId()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_setMerchantCustomerId) |  | Устанавливает идентификатор покупателя в вашей системе. |
| public | [setMetadata()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_setMetadata) |  | Устанавливает метаданные, привязанные к платежу. |
| public | [setPaymentMethodData()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_setPaymentMethodData) |  | Устанавливает объект с информацией для создания метода оплаты. |
| public | [setPaymentMethodId()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_setPaymentMethodId) |  | Устанавливает идентификатор записи о сохранённых данных покупателя. |
| public | [setPaymentOrder()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_setPaymentOrder) |  | Устанавливает платежное поручение. |
| public | [setPaymentToken()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_setPaymentToken) |  | Устанавливает одноразовый токен для проведения оплаты, сформированный YooKassa JS widget. |
| public | [setReceiver()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_setReceiver) |  | Устанавливает реквизиты получателя оплаты. |
| public | [setRecipient()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_setRecipient) |  | Устанавливает объект с информацией о получателе платежа. |
| public | [setSavePaymentMethod()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_setSavePaymentMethod) |  | Устанавливает флаг сохранения платёжных данных. Значение true инициирует создание многоразового payment_method. |
| public | [setTransfers()](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md#method_setTransfers) |  | Устанавливает данные о распределении денег — сколько и в какой магазин нужно перевести. |

---
### Details
* File: [lib/Request/Payments/CreatePaymentRequestInterface.php](../../lib/Request/Payments/CreatePaymentRequestInterface.php)
* Package: \YooKassa\Request
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |
| property |  | Получатель платежа, если задан |
| property |  | Сумма создаваемого платежа |
| property |  | Описание транзакции |
| property |  | Данные фискального чека 54-ФЗ |
| property |  | Одноразовый токен для проведения оплаты, сформированный YooKassa JS widget |
| property |  | Одноразовый токен для проведения оплаты, сформированный YooKassa JS widget |
| property |  | Идентификатор записи о сохраненных платежных данных покупателя |
| property |  | Идентификатор записи о сохраненных платежных данных покупателя |
| property |  | Данные используемые для создания метода оплаты |
| property |  | Данные используемые для создания метода оплаты |
| property |  | Способ подтверждения платежа |
| property |  | Сохранить платежные данные для последующего использования. Значение true инициирует создание многоразового payment_method |
| property |  | Сохранить платежные данные для последующего использования. Значение true инициирует создание многоразового payment_method |
| property |  | Автоматически принять поступившую оплату |
| property |  | IPv4 или IPv6-адрес покупателя. Если не указан, используется IP-адрес TCP-подключения |
| property |  | IPv4 или IPv6-адрес покупателя. Если не указан, используется IP-адрес TCP-подключения |
| property |  | Метаданные привязанные к платежу |
| property |  | Данные о сделке, в составе которой проходит платеж |
| property |  | Идентификатор покупателя в вашей системе, например электронная почта или номер телефона |
| property |  | Идентификатор покупателя в вашей системе, например электронная почта или номер телефона |
| property |  | Реквизиты получателя оплаты при пополнении электронного кошелька, банковского счета или баланса телефона |

---
## Methods
<a name="method_getRecipient" class="anchor"></a>
#### public getRecipient() : null|\YooKassa\Request\Payments\Recipient

```php
public getRecipient() : null|\YooKassa\Request\Payments\Recipient
```

**Summary**

Возвращает объект получателя платежа.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** null|\YooKassa\Request\Payments\Recipient - Объект с информацией о получателе платежа или null, если получатель не задан


<a name="method_hasRecipient" class="anchor"></a>
#### public hasRecipient() : bool

```php
public hasRecipient() : bool
```

**Summary**

Проверяет наличие получателя платежа в запросе.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - True если получатель платежа задан, false если нет


<a name="method_setRecipient" class="anchor"></a>
#### public setRecipient() : mixed

```php
public setRecipient(null|\YooKassa\Request\Payments\Recipient $recipient) : mixed
```

**Summary**

Устанавливает объект с информацией о получателе платежа.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \YooKassa\Request\Payments\Recipient</code> | recipient  | Инстанс объекта информации о получателе платежа или null |

**Returns:** mixed - 


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает сумму заказа.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма заказа


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает описание транзакции.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** string|null - Описание транзакции


<a name="method_hasDescription" class="anchor"></a>
#### public hasDescription() : bool

```php
public hasDescription() : bool
```

**Summary**

Проверяет наличие описания транзакции в создаваемом платеже.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - True если описание транзакции установлено, false если нет


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(string|null $description) : self
```

**Summary**

Устанавливает описание транзакции.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Описание транзакции |

**Returns:** self - 


<a name="method_getReceipt" class="anchor"></a>
#### public getReceipt() : null|\YooKassa\Model\Receipt\ReceiptInterface

```php
public getReceipt() : null|\YooKassa\Model\Receipt\ReceiptInterface
```

**Summary**

Возвращает чек, если он есть.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** null|\YooKassa\Model\Receipt\ReceiptInterface - Данные фискального чека 54-ФЗ или null, если чека нет


<a name="method_hasReceipt" class="anchor"></a>
#### public hasReceipt() : bool

```php
public hasReceipt() : bool
```

**Summary**

Проверяет наличие чека в создаваемом платеже.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - True если чек есть, false если нет


<a name="method_getPaymentToken" class="anchor"></a>
#### public getPaymentToken() : string|null

```php
public getPaymentToken() : string|null
```

**Summary**

Возвращает одноразовый токен для проведения оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** string|null - Одноразовый токен для проведения оплаты, сформированный YooKassa JS widget


<a name="method_hasPaymentToken" class="anchor"></a>
#### public hasPaymentToken() : bool

```php
public hasPaymentToken() : bool
```

**Summary**

Проверяет наличие одноразового токена для проведения оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - True если токен установлен, false если нет


<a name="method_setPaymentToken" class="anchor"></a>
#### public setPaymentToken() : self

```php
public setPaymentToken(string|null $payment_token) : self
```

**Summary**

Устанавливает одноразовый токен для проведения оплаты, сформированный YooKassa JS widget.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | payment_token  | Одноразовый токен для проведения оплаты |

**Returns:** self - 


<a name="method_getPaymentMethodId" class="anchor"></a>
#### public getPaymentMethodId() : string|null

```php
public getPaymentMethodId() : string|null
```

**Summary**

Устанавливает идентификатор записи платёжных данных покупателя.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** string|null - Идентификатор записи о сохраненных платежных данных покупателя


<a name="method_hasPaymentMethodId" class="anchor"></a>
#### public hasPaymentMethodId() : bool

```php
public hasPaymentMethodId() : bool
```

**Summary**

Проверяет наличие идентификатора записи о платёжных данных покупателя.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - True если идентификатор задан, false если нет


<a name="method_setPaymentMethodId" class="anchor"></a>
#### public setPaymentMethodId() : self

```php
public setPaymentMethodId(string|null $payment_method_id) : self
```

**Summary**

Устанавливает идентификатор записи о сохранённых данных покупателя.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | payment_method_id  | Идентификатор записи о сохраненных платежных данных покупателя |

**Returns:** self - 


<a name="method_getPaymentMethodData" class="anchor"></a>
#### public getPaymentMethodData() : \YooKassa\Request\Payments\PaymentData\AbstractPaymentData|null

```php
public getPaymentMethodData() : \YooKassa\Request\Payments\PaymentData\AbstractPaymentData|null
```

**Summary**

Возвращает данные для создания метода оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** \YooKassa\Request\Payments\PaymentData\AbstractPaymentData|null - Данные используемые для создания метода оплаты


<a name="method_hasPaymentMethodData" class="anchor"></a>
#### public hasPaymentMethodData() : bool

```php
public hasPaymentMethodData() : bool
```

**Summary**

Проверяет установлен ли объект с методом оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - True если объект метода оплаты установлен, false если нет


<a name="method_setPaymentMethodData" class="anchor"></a>
#### public setPaymentMethodData() : self

```php
public setPaymentMethodData(null|\YooKassa\Request\Payments\PaymentData\AbstractPaymentData $payment_method_data) : self
```

**Summary**

Устанавливает объект с информацией для создания метода оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \YooKassa\Request\Payments\PaymentData\AbstractPaymentData</code> | payment_method_data  | Объект создания метода оплаты или null |

**Returns:** self - 


<a name="method_getConfirmation" class="anchor"></a>
#### public getConfirmation() : \YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes|null

```php
public getConfirmation() : \YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes|null
```

**Summary**

Возвращает способ подтверждения платежа.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** \YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes|null - Способ подтверждения платежа


<a name="method_hasConfirmation" class="anchor"></a>
#### public hasConfirmation() : bool

```php
public hasConfirmation() : bool
```

**Summary**

Проверяет, был ли установлен способ подтверждения платежа.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - True если способ подтверждения платежа был установлен, false если нет


<a name="method_setConfirmation" class="anchor"></a>
#### public setConfirmation() : self

```php
public setConfirmation(null|array|\YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes $confirmation) : self
```

**Summary**

Устанавливает способ подтверждения платежа.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes</code> | confirmation  | Способ подтверждения платежа |

**Returns:** self - 


<a name="method_getSavePaymentMethod" class="anchor"></a>
#### public getSavePaymentMethod() : bool|null

```php
public getSavePaymentMethod() : bool|null
```

**Summary**

Возвращает флаг сохранения платёжных данных.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool|null - Флаг сохранения платёжных данных


<a name="method_hasSavePaymentMethod" class="anchor"></a>
#### public hasSavePaymentMethod() : bool

```php
public hasSavePaymentMethod() : bool
```

**Summary**

Проверяет, был ли установлен флаг сохранения платёжных данных.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - True если флыг был установлен, false если нет


<a name="method_setSavePaymentMethod" class="anchor"></a>
#### public setSavePaymentMethod() : self

```php
public setSavePaymentMethod(bool|null $save_payment_method = null) : self
```

**Summary**

Устанавливает флаг сохранения платёжных данных. Значение true инициирует создание многоразового payment_method.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool OR null</code> | save_payment_method  | Сохранить платежные данные для последующего использования |

**Returns:** self - 


<a name="method_getCapture" class="anchor"></a>
#### public getCapture() : bool

```php
public getCapture() : bool
```

**Summary**

Возвращает флаг автоматического принятия поступившей оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - True если требуется автоматически принять поступившую оплату, false если нет


<a name="method_hasCapture" class="anchor"></a>
#### public hasCapture() : bool

```php
public hasCapture() : bool
```

**Summary**

Проверяет, был ли установлен флаг автоматического приняти поступившей оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - True если флаг автоматического принятия оплаты был установлен, false если нет


<a name="method_setCapture" class="anchor"></a>
#### public setCapture() : self

```php
public setCapture(bool $capture) : self
```

**Summary**

Устанавливает флаг автоматического принятия поступившей оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool</code> | capture  | Автоматически принять поступившую оплату |

**Returns:** self - 


<a name="method_getClientIp" class="anchor"></a>
#### public getClientIp() : string|null

```php
public getClientIp() : string|null
```

**Summary**

Возвращает IPv4 или IPv6-адрес покупателя.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** string|null - IPv4 или IPv6-адрес покупателя


<a name="method_hasClientIp" class="anchor"></a>
#### public hasClientIp() : bool

```php
public hasClientIp() : bool
```

**Summary**

Проверяет, был ли установлен IPv4 или IPv6-адрес покупателя.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - True если IP адрес покупателя был установлен, false если нет


<a name="method_setClientIp" class="anchor"></a>
#### public setClientIp() : self

```php
public setClientIp(string|null $client_ip) : self
```

**Summary**

Устанавливает IP адрес покупателя.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | client_ip  | IPv4 или IPv6-адрес покупателя |

**Returns:** self - 


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает данные оплаты установленные мерчантом

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** \YooKassa\Model\Metadata|null - Метаданные привязанные к платежу


<a name="method_hasMetadata" class="anchor"></a>
#### public hasMetadata() : bool

```php
public hasMetadata() : bool
```

**Summary**

Проверяет, были ли установлены метаданные заказа.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - True если метаданные были установлены, false если нет


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : self

```php
public setMetadata(null|array|\YooKassa\Model\Metadata $metadata) : self
```

**Summary**

Устанавливает метаданные, привязанные к платежу.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\Metadata</code> | metadata  | Метаданные платежа, устанавливаемые мерчантом |

**Returns:** self - 


<a name="method_getAirline" class="anchor"></a>
#### public getAirline() : ?\YooKassa\Request\Payments\AirlineInterface

```php
public getAirline() : ?\YooKassa\Request\Payments\AirlineInterface
```

**Summary**

Возвращает данные длинной записи.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** ?\YooKassa\Request\Payments\AirlineInterface - 


<a name="method_hasAirline" class="anchor"></a>
#### public hasAirline() : bool

```php
public hasAirline() : bool
```

**Summary**

Проверяет, были ли установлены данные длинной записи.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - 


<a name="method_setAirline" class="anchor"></a>
#### public setAirline() : \YooKassa\Common\AbstractRequestInterface

```php
public setAirline(\YooKassa\Request\Payments\AirlineInterface|array|null $airline) : \YooKassa\Common\AbstractRequestInterface
```

**Summary**

Устанавливает данные авиабилетов.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Payments\AirlineInterface OR array OR null</code> | airline  | Данные авиабилетов |

**Returns:** \YooKassa\Common\AbstractRequestInterface - 


<a name="method_hasTransfers" class="anchor"></a>
#### public hasTransfers() : bool

```php
public hasTransfers() : bool
```

**Summary**

Проверяет наличие данных о распределении денег.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - 


<a name="method_getTransfers" class="anchor"></a>
#### public getTransfers() : \YooKassa\Model\Payment\TransferInterface[]|\YooKassa\Common\ListObjectInterface|null

```php
public getTransfers() : \YooKassa\Model\Payment\TransferInterface[]|\YooKassa\Common\ListObjectInterface|null
```

**Summary**

Возвращает данные о распределении денег — сколько и в какой магазин нужно перевести.

**Description**

Присутствует, если вы используете решение ЮKassa для платформ.
(https://yookassa.ru/developers/special-solutions/checkout-for-platforms/basics).

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** \YooKassa\Model\Payment\TransferInterface[]|\YooKassa\Common\ListObjectInterface|null - Данные о распределении денег


<a name="method_setTransfers" class="anchor"></a>
#### public setTransfers() : self

```php
public setTransfers(\YooKassa\Common\ListObjectInterface|array|null $transfers = null) : self
```

**Summary**

Устанавливает данные о распределении денег — сколько и в какой магазин нужно перевести.

**Description**

Присутствует, если вы используете решение ЮKassa для платформ.
(https://yookassa.ru/developers/special-solutions/checkout-for-platforms/basics).

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Common\ListObjectInterface OR array OR null</code> | transfers  | Массив распределения денег |

**Returns:** self - 


<a name="method_getDeal" class="anchor"></a>
#### public getDeal() : \YooKassa\Model\Deal\PaymentDealInfo|null

```php
public getDeal() : \YooKassa\Model\Deal\PaymentDealInfo|null
```

**Summary**

Возвращает данные о сделке, в составе которой проходит платеж.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** \YooKassa\Model\Deal\PaymentDealInfo|null - Данные о сделке, в составе которой проходит платеж


<a name="method_hasDeal" class="anchor"></a>
#### public hasDeal() : bool

```php
public hasDeal() : bool
```

**Summary**

Проверяет, были ли установлены данные о сделке.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - True если данные о сделке были установлены, false если нет


<a name="method_setDeal" class="anchor"></a>
#### public setDeal() : self

```php
public setDeal(null|array|\YooKassa\Model\Deal\PaymentDealInfo $deal) : self
```

**Summary**

Устанавливает данные о сделке, в составе которой проходит платеж.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\Deal\PaymentDealInfo</code> | deal  | Данные о сделке, в составе которой проходит платеж |

**Returns:** self - 


<a name="method_getMerchantCustomerId" class="anchor"></a>
#### public getMerchantCustomerId() : string|null

```php
public getMerchantCustomerId() : string|null
```

**Summary**

Возвращает идентификатор покупателя в вашей системе.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** string|null - Идентификатор покупателя в вашей системе


<a name="method_hasMerchantCustomerId" class="anchor"></a>
#### public hasMerchantCustomerId() : bool

```php
public hasMerchantCustomerId() : bool
```

**Summary**

Проверяет, был ли установлен идентификатор покупателя в вашей системе.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - True если идентификатор покупателя был установлен, false если нет


<a name="method_setMerchantCustomerId" class="anchor"></a>
#### public setMerchantCustomerId() : self

```php
public setMerchantCustomerId(string|null $merchant_customer_id) : self
```

**Summary**

Устанавливает идентификатор покупателя в вашей системе.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | merchant_customer_id  | Идентификатор покупателя в вашей системе, например электронная почта или номер телефона. Не более 200 символов |

**Returns:** self - 


<a name="method_getPaymentOrder" class="anchor"></a>
#### public getPaymentOrder() : \YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder|null

```php
public getPaymentOrder() : \YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder|null
```

**Summary**

Возвращает платежное поручение.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** \YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder|null - Платежное поручение — распоряжение на перевод банку для оплаты жилищно-коммунальных услуг (ЖКУ), сведения о платеже для регистрации в ГИС ЖКХ.


<a name="method_hasPaymentOrder" class="anchor"></a>
#### public hasPaymentOrder() : bool

```php
public hasPaymentOrder() : bool
```

**Summary**

Проверяет, было ли установлено платежное поручение.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - True если платежное поручение было установлены, false если нет


<a name="method_setPaymentOrder" class="anchor"></a>
#### public setPaymentOrder() : self

```php
public setPaymentOrder(\YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder|array|null $payment_order = null) : self
```

**Summary**

Устанавливает платежное поручение.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder OR array OR null</code> | payment_order  | Платежное поручение — распоряжение на перевод банку для оплаты жилищно-коммунальных услуг (ЖКУ), сведения о платеже для регистрации в ГИС ЖКХ. |

**Returns:** self - 


<a name="method_getReceiver" class="anchor"></a>
#### public getReceiver() : null|\YooKassa\Request\Payments\ReceiverData\AbstractReceiver

```php
public getReceiver() : null|\YooKassa\Request\Payments\ReceiverData\AbstractReceiver
```

**Summary**

Возвращает реквизиты получателя оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** null|\YooKassa\Request\Payments\ReceiverData\AbstractReceiver - Реквизиты получателя оплаты при пополнении электронного кошелька, банковского счета или баланса телефона.


<a name="method_hasReceiver" class="anchor"></a>
#### public hasReceiver() : bool

```php
public hasReceiver() : bool
```

**Summary**

Проверяет, были ли установлены реквизиты получателя оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

**Returns:** bool - True если реквизиты получателя оплаты были установлены, false если нет


<a name="method_setReceiver" class="anchor"></a>
#### public setReceiver() : self

```php
public setReceiver(null|array|\YooKassa\Request\Payments\ReceiverData\AbstractReceiver $receiver) : self
```

**Summary**

Устанавливает реквизиты получателя оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestInterface](../classes/YooKassa-Request-Payments-CreatePaymentRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Request\Payments\ReceiverData\AbstractReceiver</code> | receiver  | Реквизиты получателя оплаты при пополнении электронного кошелька, банковского счета или баланса телефона. |

**Returns:** self - 




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
# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Invoices\PaymentData
### Namespace: [\YooKassa\Request\Invoices](../namespaces/yookassa-request-invoices.md)
---
**Summary:**

Класс, представляющий модель PaymentData.

**Description:**

Данные для проведения платежа по выставленному счету.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$amount](../classes/YooKassa-Request-Invoices-PaymentData.md#property_amount) |  | Сумма платежа. Должна укладываться в [лимиты](https://yookassa.ru/docs/support/payments/limits). |
| public | [$capture](../classes/YooKassa-Request-Invoices-PaymentData.md#property_capture) |  | [Автоматический прием](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#capture-true) поступившего платежа. |
| public | [$client_ip](../classes/YooKassa-Request-Invoices-PaymentData.md#property_client_ip) |  | IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения. |
| public | [$clientIp](../classes/YooKassa-Request-Invoices-PaymentData.md#property_clientIp) |  | IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения. |
| public | [$description](../classes/YooKassa-Request-Invoices-PaymentData.md#property_description) |  | Описание транзакции (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь — при оплате. Например: «Оплата заказа № 72 для user@yoomoney.ru». |
| public | [$metadata](../classes/YooKassa-Request-Invoices-PaymentData.md#property_metadata) |  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. |
| public | [$receipt](../classes/YooKassa-Request-Invoices-PaymentData.md#property_receipt) |  | Данные для формирования чека. |
| public | [$recipient](../classes/YooKassa-Request-Invoices-PaymentData.md#property_recipient) |  | Получатель платежа. Нужен, если вы разделяете потоки платежей в рамках одного аккаунта или создаете платеж в адрес другого аккаунта. |
| public | [$save_payment_method](../classes/YooKassa-Request-Invoices-PaymentData.md#property_save_payment_method) |  | Сохранение платежных данных для проведения [автоплатежей](https://yookassa.ru/developers/payment-acceptance/scenario-extensions/recurring-payments). |
| public | [$savePaymentMethod](../classes/YooKassa-Request-Invoices-PaymentData.md#property_savePaymentMethod) |  | Сохранение платежных данных для проведения [автоплатежей](https://yookassa.ru/developers/payment-acceptance/scenario-extensions/recurring-payments). |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getAmount()](../classes/YooKassa-Request-Invoices-PaymentData.md#method_getAmount) |  | Возвращает amount. |
| public | [getCapture()](../classes/YooKassa-Request-Invoices-PaymentData.md#method_getCapture) |  | Возвращает capture. |
| public | [getClientIp()](../classes/YooKassa-Request-Invoices-PaymentData.md#method_getClientIp) |  | Возвращает client_ip. |
| public | [getDescription()](../classes/YooKassa-Request-Invoices-PaymentData.md#method_getDescription) |  | Возвращает description. |
| public | [getMetadata()](../classes/YooKassa-Request-Invoices-PaymentData.md#method_getMetadata) |  | Возвращает metadata. |
| public | [getReceipt()](../classes/YooKassa-Request-Invoices-PaymentData.md#method_getReceipt) |  | Возвращает receipt. |
| public | [getRecipient()](../classes/YooKassa-Request-Invoices-PaymentData.md#method_getRecipient) |  | Возвращает recipient. |
| public | [getSavePaymentMethod()](../classes/YooKassa-Request-Invoices-PaymentData.md#method_getSavePaymentMethod) |  | Возвращает save_payment_method. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setAmount()](../classes/YooKassa-Request-Invoices-PaymentData.md#method_setAmount) |  | Устанавливает amount. |
| public | [setCapture()](../classes/YooKassa-Request-Invoices-PaymentData.md#method_setCapture) |  | Устанавливает capture. |
| public | [setClientIp()](../classes/YooKassa-Request-Invoices-PaymentData.md#method_setClientIp) |  | Устанавливает client_ip. |
| public | [setDescription()](../classes/YooKassa-Request-Invoices-PaymentData.md#method_setDescription) |  | Устанавливает description. |
| public | [setMetadata()](../classes/YooKassa-Request-Invoices-PaymentData.md#method_setMetadata) |  | Устанавливает metadata. |
| public | [setReceipt()](../classes/YooKassa-Request-Invoices-PaymentData.md#method_setReceipt) |  | Устанавливает receipt. |
| public | [setRecipient()](../classes/YooKassa-Request-Invoices-PaymentData.md#method_setRecipient) |  | Устанавливает recipient. |
| public | [setSavePaymentMethod()](../classes/YooKassa-Request-Invoices-PaymentData.md#method_setSavePaymentMethod) |  | Устанавливает save_payment_method. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Invoices/PaymentData.php](../../lib/Request/Invoices/PaymentData.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Request\Invoices\PaymentData

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Properties
<a name="property_amount"></a>
#### public $amount : \YooKassa\Model\AmountInterface
---
***Description***

Сумма платежа. Должна укладываться в [лимиты](https://yookassa.ru/docs/support/payments/limits).

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_capture"></a>
#### public $capture : bool
---
***Description***

[Автоматический прием](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#capture-true) поступившего платежа.

**Type:** <a href="../bool"><abbr title="bool">bool</abbr></a>

**Details:**


<a name="property_client_ip"></a>
#### public $client_ip : string
---
***Description***

IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_clientIp"></a>
#### public $clientIp : string
---
***Description***

IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_description"></a>
#### public $description : string
---
***Description***

Описание транзакции (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь — при оплате. Например: «Оплата заказа № 72 для user@yoomoney.ru».

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_metadata"></a>
#### public $metadata : array
---
***Description***

Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa.

**Type:** <a href="../array"><abbr title="array">array</abbr></a>

**Details:**


<a name="property_receipt"></a>
#### public $receipt : \YooKassa\Model\Receipt\ReceiptInterface
---
***Description***

Данные для формирования чека.

**Type:** <a href="../classes/YooKassa-Model-Receipt-ReceiptInterface.html"><abbr title="\YooKassa\Model\Receipt\ReceiptInterface">ReceiptInterface</abbr></a>

**Details:**


<a name="property_recipient"></a>
#### public $recipient : \YooKassa\Model\Payment\RecipientInterface
---
***Description***

Получатель платежа. Нужен, если вы разделяете потоки платежей в рамках одного аккаунта или создаете платеж в адрес другого аккаунта.

**Type:** <a href="../classes/YooKassa-Model-Payment-RecipientInterface.html"><abbr title="\YooKassa\Model\Payment\RecipientInterface">RecipientInterface</abbr></a>

**Details:**


<a name="property_save_payment_method"></a>
#### public $save_payment_method : bool
---
***Description***

Сохранение платежных данных для проведения [автоплатежей](https://yookassa.ru/developers/payment-acceptance/scenario-extensions/recurring-payments).

**Type:** <a href="../bool"><abbr title="bool">bool</abbr></a>

**Details:**


<a name="property_savePaymentMethod"></a>
#### public $savePaymentMethod : bool
---
***Description***

Сохранение платежных данных для проведения [автоплатежей](https://yookassa.ru/developers/payment-acceptance/scenario-extensions/recurring-payments).

**Type:** <a href="../bool"><abbr title="bool">bool</abbr></a>

**Details:**



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(array|null $data = []) : mixed
```

**Summary**

AbstractObject constructor.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR null</code> | data  |  |

**Returns:** mixed - 


<a name="method___get" class="anchor"></a>
#### public __get() : mixed

```php
public __get(string $propertyName) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method___isset" class="anchor"></a>
#### public __isset() : bool

```php
public __isset(string $propertyName) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method___set" class="anchor"></a>
#### public __set() : void

```php
public __set(string $propertyName, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method___unset" class="anchor"></a>
#### public __unset() : void

```php
public __unset(string $propertyName) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_fromArray" class="anchor"></a>
#### public fromArray() : void

```php
public fromArray(array|\Traversable $sourceArray) : void
```

**Summary**

Устанавливает значения свойств текущего объекта из массива.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \Traversable</code> | sourceArray  | Ассоциативный массив с настройками |

**Returns:** void - 


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает amount.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\PaymentData](../classes/YooKassa-Request-Invoices-PaymentData.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма платежа


<a name="method_getCapture" class="anchor"></a>
#### public getCapture() : bool|null

```php
public getCapture() : bool|null
```

**Summary**

Возвращает capture.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\PaymentData](../classes/YooKassa-Request-Invoices-PaymentData.md)

**Returns:** bool|null - Автоматический прием поступившего платежа


<a name="method_getClientIp" class="anchor"></a>
#### public getClientIp() : string|null

```php
public getClientIp() : string|null
```

**Summary**

Возвращает client_ip.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\PaymentData](../classes/YooKassa-Request-Invoices-PaymentData.md)

**Returns:** string|null - IPv4 или IPv6-адрес пользователя


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает description.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\PaymentData](../classes/YooKassa-Request-Invoices-PaymentData.md)

**Returns:** string|null - Описание транзакции


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает metadata.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\PaymentData](../classes/YooKassa-Request-Invoices-PaymentData.md)

**Returns:** \YooKassa\Model\Metadata|null - Любые дополнительные данные


<a name="method_getReceipt" class="anchor"></a>
#### public getReceipt() : \YooKassa\Model\Receipt\ReceiptInterface|null

```php
public getReceipt() : \YooKassa\Model\Receipt\ReceiptInterface|null
```

**Summary**

Возвращает receipt.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\PaymentData](../classes/YooKassa-Request-Invoices-PaymentData.md)

**Returns:** \YooKassa\Model\Receipt\ReceiptInterface|null - Данные для формирования чека


<a name="method_getRecipient" class="anchor"></a>
#### public getRecipient() : \YooKassa\Model\Payment\Recipient|null

```php
public getRecipient() : \YooKassa\Model\Payment\Recipient|null
```

**Summary**

Возвращает recipient.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\PaymentData](../classes/YooKassa-Request-Invoices-PaymentData.md)

**Returns:** \YooKassa\Model\Payment\Recipient|null - Получатель платежа


<a name="method_getSavePaymentMethod" class="anchor"></a>
#### public getSavePaymentMethod() : bool|null

```php
public getSavePaymentMethod() : bool|null
```

**Summary**

Возвращает save_payment_method.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\PaymentData](../classes/YooKassa-Request-Invoices-PaymentData.md)

**Returns:** bool|null - Сохранение платежных данных для проведения автоплатежей


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_jsonSerialize" class="anchor"></a>
#### public jsonSerialize() : array

```php
public jsonSerialize() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_offsetExists" class="anchor"></a>
#### public offsetExists() : bool

```php
public offsetExists(string $offset) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method_offsetGet" class="anchor"></a>
#### public offsetGet() : mixed

```php
public offsetGet(string $offset) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method_offsetSet" class="anchor"></a>
#### public offsetSet() : void

```php
public offsetSet(string $offset, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method_offsetUnset" class="anchor"></a>
#### public offsetUnset() : void

```php
public offsetUnset(string $offset) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_setAmount" class="anchor"></a>
#### public setAmount() : self

```php
public setAmount(\YooKassa\Model\AmountInterface|array|null $amount = null) : self
```

**Summary**

Устанавливает amount.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\PaymentData](../classes/YooKassa-Request-Invoices-PaymentData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | amount  | Сумма платежа |

**Returns:** self - 


<a name="method_setCapture" class="anchor"></a>
#### public setCapture() : self

```php
public setCapture(bool|array|null $capture = null) : self
```

**Summary**

Устанавливает capture.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\PaymentData](../classes/YooKassa-Request-Invoices-PaymentData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool OR array OR null</code> | capture  | Автоматический прием поступившего платежа |

**Returns:** self - 


<a name="method_setClientIp" class="anchor"></a>
#### public setClientIp() : self

```php
public setClientIp(string|null $client_ip = null) : self
```

**Summary**

Устанавливает client_ip.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\PaymentData](../classes/YooKassa-Request-Invoices-PaymentData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | client_ip  | IPv4 или IPv6-адрес пользователя |

**Returns:** self - 


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(string|null $description = null) : self
```

**Summary**

Устанавливает description.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\PaymentData](../classes/YooKassa-Request-Invoices-PaymentData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Описание транзакции |

**Returns:** self - 


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : self

```php
public setMetadata(\YooKassa\Model\Metadata|array|null $metadata = null) : self
```

**Summary**

Устанавливает metadata.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\PaymentData](../classes/YooKassa-Request-Invoices-PaymentData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Metadata OR array OR null</code> | metadata  | Любые дополнительные данные |

**Returns:** self - 


<a name="method_setReceipt" class="anchor"></a>
#### public setReceipt() : self

```php
public setReceipt(\YooKassa\Model\Receipt\ReceiptInterface|array|null $receipt = null) : self
```

**Summary**

Устанавливает receipt.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\PaymentData](../classes/YooKassa-Request-Invoices-PaymentData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Receipt\ReceiptInterface OR array OR null</code> | receipt  | Данные для формирования чека |

**Returns:** self - 


<a name="method_setRecipient" class="anchor"></a>
#### public setRecipient() : self

```php
public setRecipient(\YooKassa\Model\Payment\Recipient|array|null $recipient = null) : self
```

**Summary**

Устанавливает recipient.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\PaymentData](../classes/YooKassa-Request-Invoices-PaymentData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Payment\Recipient OR array OR null</code> | recipient  | Получатель платежа |

**Returns:** self - 


<a name="method_setSavePaymentMethod" class="anchor"></a>
#### public setSavePaymentMethod() : self

```php
public setSavePaymentMethod(bool|array|null $save_payment_method = null) : self
```

**Summary**

Устанавливает save_payment_method.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\PaymentData](../classes/YooKassa-Request-Invoices-PaymentData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool OR array OR null</code> | save_payment_method  | Сохранение платежных данных для проведения автоплатежей |

**Returns:** self - 


<a name="method_toArray" class="anchor"></a>
#### public toArray() : array

```php
public toArray() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации
Является алиасом метода AbstractObject::jsonSerialize().

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_getUnknownProperties" class="anchor"></a>
#### protected getUnknownProperties() : array

```php
protected getUnknownProperties() : array
```

**Summary**

Возвращает массив свойств которые не существуют, но были заданы у объекта.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив с не существующими у текущего объекта свойствами


<a name="method_validatePropertyValue" class="anchor"></a>
#### protected validatePropertyValue() : mixed

```php
protected validatePropertyValue(string $propertyName, mixed $propertyValue) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  |  |
| <code lang="php">mixed</code> | propertyValue  |  |

**Returns:** mixed - 



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
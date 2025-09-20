# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\AbstractPaymentRequest
### Namespace: [\YooKassa\Request\Payments](../namespaces/yookassa-request-payments.md)
---
**Summary:**

Класс, представляющий модель AbstractPaymentRequest.

**Description:**

Базовый класс объекта запроса к API.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$airline](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#property_airline) |  | Объект с данными для продажи авиабилетов |
| public | [$amount](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#property_amount) |  | Сумма |
| public | [$receipt](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#property_receipt) |  | Данные фискального чека 54-ФЗ |
| public | [$transfers](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#property_transfers) |  | Данные о распределении платежа между магазинами |
| protected | [$_airline](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#property__airline) |  |  |
| protected | [$_amount](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#property__amount) |  |  |
| protected | [$_receipt](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#property__receipt) |  |  |
| protected | [$_transfers](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#property__transfers) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [clearValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_clearValidationError) |  | Очищает статус валидации текущего запроса. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getAirline()](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#method_getAirline) |  | Возвращает данные авиабилетов. |
| public | [getAmount()](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#method_getAmount) |  | Возвращает сумму оплаты. |
| public | [getLastValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_getLastValidationError) |  | Возвращает последнюю ошибку валидации. |
| public | [getReceipt()](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#method_getReceipt) |  | Возвращает чек, если он есть. |
| public | [getTransfers()](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#method_getTransfers) |  | Возвращает данные о распределении денег — сколько и в какой магазин нужно перевести. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [hasAirline()](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#method_hasAirline) |  | Проверяет, были ли установлены данные авиабилетов. |
| public | [hasAmount()](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#method_hasAmount) |  | Проверяет, была ли установлена сумма оплаты. |
| public | [hasReceipt()](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#method_hasReceipt) |  | Проверяет наличие чека. |
| public | [hasTransfers()](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#method_hasTransfers) |  | Проверяет наличие данных о распределении денег. |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [removeReceipt()](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#method_removeReceipt) |  | Удаляет чек из запроса. |
| public | [setAirline()](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#method_setAirline) |  | Устанавливает данные авиабилетов. |
| public | [setAmount()](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#method_setAmount) |  | Устанавливает сумму оплаты. |
| public | [setReceipt()](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#method_setReceipt) |  | Устанавливает чек. |
| public | [setTransfers()](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#method_setTransfers) |  | Устанавливает transfers (массив распределения денег между магазинами). |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| public | [validate()](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md#method_validate) |  | Валидирует объект запроса. |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [setValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_setValidationError) |  | Устанавливает ошибку валидации. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Payments/AbstractPaymentRequest.php](../../lib/Request/Payments/AbstractPaymentRequest.php)
* Package: YooKassa\Request
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)
  * \YooKassa\Request\Payments\AbstractPaymentRequest

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
<a name="property_airline"></a>
#### public $airline : \YooKassa\Request\Payments\AirlineInterface|null
---
***Description***

Объект с данными для продажи авиабилетов

**Type:** <a href="../\YooKassa\Request\Payments\AirlineInterface|null"><abbr title="\YooKassa\Request\Payments\AirlineInterface|null">AirlineInterface|null</abbr></a>

**Details:**


<a name="property_amount"></a>
#### public $amount : \YooKassa\Model\AmountInterface
---
***Description***

Сумма

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_receipt"></a>
#### public $receipt : \YooKassa\Model\Receipt\ReceiptInterface|null
---
***Description***

Данные фискального чека 54-ФЗ

**Type:** <a href="../\YooKassa\Model\Receipt\ReceiptInterface|null"><abbr title="\YooKassa\Model\Receipt\ReceiptInterface|null">ReceiptInterface|null</abbr></a>

**Details:**


<a name="property_transfers"></a>
#### public $transfers : \YooKassa\Common\ListObjectInterface|\YooKassa\Request\Payments\TransferDataInterface[]|null
---
***Description***

Данные о распределении платежа между магазинами

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Request\Payments\TransferDataInterface[]|null"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Request\Payments\TransferDataInterface[]|null">TransferDataInterface[]|null</abbr></a>

**Details:**


<a name="property__airline"></a>
#### protected $_airline : ?\YooKassa\Request\Payments\AirlineInterface
---
**Type:** <a href="../?\YooKassa\Request\Payments\AirlineInterface"><abbr title="?\YooKassa\Request\Payments\AirlineInterface">AirlineInterface</abbr></a>
Объект с данными для продажи авиабилетов
**Details:**


<a name="property__amount"></a>
#### protected $_amount : ?\YooKassa\Model\AmountInterface
---
**Type:** <a href="../?\YooKassa\Model\AmountInterface"><abbr title="?\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>
Сумма оплаты
**Details:**


<a name="property__receipt"></a>
#### protected $_receipt : ?\YooKassa\Model\Receipt\ReceiptInterface
---
**Type:** <a href="../?\YooKassa\Model\Receipt\ReceiptInterface"><abbr title="?\YooKassa\Model\Receipt\ReceiptInterface">ReceiptInterface</abbr></a>
Данные фискального чека 54-ФЗ
**Details:**


<a name="property__transfers"></a>
#### protected $_transfers : ?\YooKassa\Common\ListObject
---
**Type:** <a href="../?\YooKassa\Common\ListObject"><abbr title="?\YooKassa\Common\ListObject">ListObject</abbr></a>

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


<a name="method_clearValidationError" class="anchor"></a>
#### public clearValidationError() : void

```php
public clearValidationError() : void
```

**Summary**

Очищает статус валидации текущего запроса.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)

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


<a name="method_getAirline" class="anchor"></a>
#### public getAirline() : null|\YooKassa\Request\Payments\AirlineInterface

```php
public getAirline() : null|\YooKassa\Request\Payments\AirlineInterface
```

**Summary**

Возвращает данные авиабилетов.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequest](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md)

**Returns:** null|\YooKassa\Request\Payments\AirlineInterface - Данные авиабилетов


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает сумму оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequest](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма оплаты


<a name="method_getLastValidationError" class="anchor"></a>
#### public getLastValidationError() : string|null

```php
public getLastValidationError() : string|null
```

**Summary**

Возвращает последнюю ошибку валидации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)

**Returns:** string|null - Последняя произошедшая ошибка валидации


<a name="method_getReceipt" class="anchor"></a>
#### public getReceipt() : null|\YooKassa\Model\Receipt\ReceiptInterface

```php
public getReceipt() : null|\YooKassa\Model\Receipt\ReceiptInterface
```

**Summary**

Возвращает чек, если он есть.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequest](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md)

**Returns:** null|\YooKassa\Model\Receipt\ReceiptInterface - Данные фискального чека 54-ФЗ или null, если чека нет


<a name="method_getTransfers" class="anchor"></a>
#### public getTransfers() : \YooKassa\Request\Payments\TransferDataInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getTransfers() : \YooKassa\Request\Payments\TransferDataInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает данные о распределении денег — сколько и в какой магазин нужно перевести.

**Description**

Присутствует, если вы используете решение ЮKassa для платформ.
(https://yookassa.ru/developers/special-solutions/checkout-for-platforms/basics).

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequest](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md)

**Returns:** \YooKassa\Request\Payments\TransferDataInterface[]|\YooKassa\Common\ListObjectInterface - Данные о распределении денег


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_hasAirline" class="anchor"></a>
#### public hasAirline() : bool

```php
public hasAirline() : bool
```

**Summary**

Проверяет, были ли установлены данные авиабилетов.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequest](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md)

**Returns:** bool - 


<a name="method_hasAmount" class="anchor"></a>
#### public hasAmount() : bool

```php
public hasAmount() : bool
```

**Summary**

Проверяет, была ли установлена сумма оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequest](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md)

**Returns:** bool - True если сумма оплаты была установлена, false если нет


<a name="method_hasReceipt" class="anchor"></a>
#### public hasReceipt() : bool

```php
public hasReceipt() : bool
```

**Summary**

Проверяет наличие чека.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequest](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md)

**Returns:** bool - True если чек есть, false если нет


<a name="method_hasTransfers" class="anchor"></a>
#### public hasTransfers() : bool

```php
public hasTransfers() : bool
```

**Summary**

Проверяет наличие данных о распределении денег.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequest](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md)

**Returns:** bool - 


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


<a name="method_removeReceipt" class="anchor"></a>
#### public removeReceipt() : self

```php
public removeReceipt() : self
```

**Summary**

Удаляет чек из запроса.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequest](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md)

**Returns:** self - 


<a name="method_setAirline" class="anchor"></a>
#### public setAirline() : \YooKassa\Common\AbstractRequestInterface

```php
public setAirline(\YooKassa\Request\Payments\AirlineInterface|array|null $airline) : \YooKassa\Common\AbstractRequestInterface
```

**Summary**

Устанавливает данные авиабилетов.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequest](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Payments\AirlineInterface OR array OR null</code> | airline  | Данные авиабилетов |

**Returns:** \YooKassa\Common\AbstractRequestInterface - 


<a name="method_setAmount" class="anchor"></a>
#### public setAmount() : self

```php
public setAmount(\YooKassa\Model\AmountInterface|array|string $amount = null) : self
```

**Summary**

Устанавливает сумму оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequest](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR string</code> | amount  | Сумма оплаты |

**Returns:** self - 


<a name="method_setReceipt" class="anchor"></a>
#### public setReceipt() : self

```php
public setReceipt(null|array|\YooKassa\Model\Receipt\ReceiptInterface $receipt) : self
```

**Summary**

Устанавливает чек.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequest](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\Receipt\ReceiptInterface</code> | receipt  | Инстанс чека или null для удаления информации о чеке |

**Returns:** self - 


<a name="method_setTransfers" class="anchor"></a>
#### public setTransfers() : self

```php
public setTransfers(\YooKassa\Common\ListObjectInterface|array|null $transfers = null) : self
```

**Summary**

Устанавливает transfers (массив распределения денег между магазинами).

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequest](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Common\ListObjectInterface OR array OR null</code> | transfers  | Массив распределения денег |

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


<a name="method_validate" class="anchor"></a>
#### public validate() : bool

```php
public validate() : bool
```

**Summary**

Валидирует объект запроса.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequest](../classes/YooKassa-Request-Payments-AbstractPaymentRequest.md)

**Returns:** bool - True если запрос валиден и его можно отправить в API, false если нет


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


<a name="method_setValidationError" class="anchor"></a>
#### protected setValidationError() : void

```php
protected setValidationError(string $value) : void
```

**Summary**

Устанавливает ошибку валидации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | value  | Ошибка, произошедшая при валидации объекта |

**Returns:** void - 


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
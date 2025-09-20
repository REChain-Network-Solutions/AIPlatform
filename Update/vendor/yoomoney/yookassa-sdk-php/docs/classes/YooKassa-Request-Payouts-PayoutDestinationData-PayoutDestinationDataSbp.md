# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payouts\PayoutDestinationData\PayoutDestinationDataSbp
### Namespace: [\YooKassa\Request\Payouts\PayoutDestinationData](../namespaces/yookassa-request-payouts-payoutdestinationdata.md)
---
**Summary:**

Класс, представляющий модель PayoutDestinationDataSbp.

**Description:**

Метод оплаты, при оплате через СБП.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$bank_id](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataSbp.md#property_bank_id) |  | Идентификатор выбранного участника СБП — банка или платежного сервиса, подключенного к сервису |
| public | [$bankId](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataSbp.md#property_bankId) |  | Идентификатор выбранного участника СБП — банка или платежного сервиса, подключенного к сервису |
| public | [$phone](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataSbp.md#property_phone) |  | Телефон, к которому привязан счет получателя выплаты в системе участника СБП |
| public | [$type](../classes/YooKassa-Request-Payouts-PayoutDestinationData-AbstractPayoutDestinationData.md#property_type) |  | Тип метода оплаты |
| public | [$type](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md#property_type) |  | Тип метода оплаты |
| protected | [$_type](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md#property__type) |  | Тип метода оплаты |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataSbp.md#method___construct) |  | Конструктор PayoutDestinationDataSbp. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getBankId()](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataSbp.md#method_getBankId) |  | Возвращает идентификатор выбранного участника СБП. |
| public | [getPhone()](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataSbp.md#method_getPhone) |  | Возвращает телефон, к которому привязан счет получателя выплаты в системе участника СБП. |
| public | [getType()](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md#method_getType) |  | Возвращает тип метода оплаты. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setBankId()](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataSbp.md#method_setBankId) |  | Устанавливает идентификатор выбранного участника СБП. |
| public | [setPhone()](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataSbp.md#method_setPhone) |  | Устанавливает телефон, к которому привязан счет получателя выплаты в системе участника СБП. |
| public | [setType()](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md#method_setType) |  | Устанавливает тип метода оплаты. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Payouts/PayoutDestinationData/PayoutDestinationDataSbp.php](../../lib/Request/Payouts/PayoutDestinationData/PayoutDestinationDataSbp.php)
* Package: YooKassa\Request
* Class Hierarchy:   
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Model\Payout\AbstractPayoutDestination](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md)
  * [\YooKassa\Request\Payouts\PayoutDestinationData\AbstractPayoutDestinationData](../classes/YooKassa-Request-Payouts-PayoutDestinationData-AbstractPayoutDestinationData.md)
  * \YooKassa\Request\Payouts\PayoutDestinationData\PayoutDestinationDataSbp

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
<a name="property_bank_id"></a>
#### public $bank_id : string
---
***Description***

Идентификатор выбранного участника СБП — банка или платежного сервиса, подключенного к сервису

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_bankId"></a>
#### public $bankId : string
---
***Description***

Идентификатор выбранного участника СБП — банка или платежного сервиса, подключенного к сервису

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_phone"></a>
#### public $phone : string
---
***Description***

Телефон, к которому привязан счет получателя выплаты в системе участника СБП

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Тип метода оплаты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\PayoutDestinationData\AbstractPayoutDestinationData](../classes/YooKassa-Request-Payouts-PayoutDestinationData-AbstractPayoutDestinationData.md)


<a name="property_type"></a>
#### public $type : string
---
***Description***

Тип метода оплаты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Payout\AbstractPayoutDestination](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md)


<a name="property__type"></a>
#### protected $_type : ?string
---
**Summary**

Тип метода оплаты

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Payout\AbstractPayoutDestination](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md)



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(array|null $data = []) : mixed
```

**Summary**

Конструктор PayoutDestinationDataSbp.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\PayoutDestinationData\PayoutDestinationDataSbp](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataSbp.md)

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


<a name="method_getBankId" class="anchor"></a>
#### public getBankId() : string|null

```php
public getBankId() : string|null
```

**Summary**

Возвращает идентификатор выбранного участника СБП.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\PayoutDestinationData\PayoutDestinationDataSbp](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataSbp.md)

**Returns:** string|null - Идентификатор выбранного участника СБП


<a name="method_getPhone" class="anchor"></a>
#### public getPhone() : string|null

```php
public getPhone() : string|null
```

**Summary**

Возвращает телефон, к которому привязан счет получателя выплаты в системе участника СБП.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\PayoutDestinationData\PayoutDestinationDataSbp](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataSbp.md)

**Returns:** string|null - Телефон, к которому привязан счет получателя выплаты в системе участника СБП


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает тип метода оплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\AbstractPayoutDestination](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md)

**Returns:** string|null - Тип метода оплаты


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


<a name="method_setBankId" class="anchor"></a>
#### public setBankId() : self

```php
public setBankId(string|null $bank_id = null) : self
```

**Summary**

Устанавливает идентификатор выбранного участника СБП.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\PayoutDestinationData\PayoutDestinationDataSbp](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataSbp.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | bank_id  | Идентификатор выбранного участника СБП |

**Returns:** self - 


<a name="method_setPhone" class="anchor"></a>
#### public setPhone() : self

```php
public setPhone(string|null $phone = null) : self
```

**Summary**

Устанавливает телефон, к которому привязан счет получателя выплаты в системе участника СБП.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\PayoutDestinationData\PayoutDestinationDataSbp](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataSbp.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | phone  | Телефон, к которому привязан счет получателя выплаты в системе участника СБП |

**Returns:** self - 


<a name="method_setType" class="anchor"></a>
#### public setType() : self

```php
public setType(string|null $type = null) : self
```

**Summary**

Устанавливает тип метода оплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\AbstractPayoutDestination](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип метода оплаты |

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
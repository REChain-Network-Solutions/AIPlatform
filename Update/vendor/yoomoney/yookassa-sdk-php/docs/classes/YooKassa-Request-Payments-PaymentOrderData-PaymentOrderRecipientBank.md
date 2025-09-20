# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank
### Namespace: [\YooKassa\Request\Payments\PaymentOrderData](../namespaces/yookassa-request-payments-paymentorderdata.md)
---
**Summary:**

Класс, представляющий модель PaymentOrderRecipientBank.


---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$account](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md#property_account) |  | Счет получателя в банке. |
| public | [$bic](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md#property_bic) |  | БИК банка получателя. |
| public | [$correspondent_account](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md#property_correspondent_account) |  | Корреспондентский счет банка получателя. |
| public | [$name](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md#property_name) |  | Название банка получателя. |

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
| public | [getAccount()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md#method_getAccount) |  | Возвращает account. |
| public | [getBic()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md#method_getBic) |  | Возвращает bic. |
| public | [getCorrespondentAccount()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md#method_getCorrespondentAccount) |  | Возвращает correspondent_account. |
| public | [getName()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md#method_getName) |  | Возвращает name. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setAccount()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md#method_setAccount) |  | Устанавливает account. |
| public | [setBic()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md#method_setBic) |  | Устанавливает bic. |
| public | [setCorrespondentAccount()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md#method_setCorrespondentAccount) |  | Устанавливает correspondent_account. |
| public | [setName()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md#method_setName) |  | Устанавливает name. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Payments/PaymentOrderData/PaymentOrderRecipientBank.php](../../lib/Request/Payments/PaymentOrderData/PaymentOrderRecipientBank.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank

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
<a name="property_account"></a>
#### public $account : string
---
***Description***

Счет получателя в банке.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_bic"></a>
#### public $bic : string
---
***Description***

БИК банка получателя.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_correspondent_account"></a>
#### public $correspondent_account : string
---
***Description***

Корреспондентский счет банка получателя.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_name"></a>
#### public $name : string
---
***Description***

Название банка получателя.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

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


<a name="method_getAccount" class="anchor"></a>
#### public getAccount() : string|null

```php
public getAccount() : string|null
```

**Summary**

Возвращает account.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md)

**Returns:** string|null - 


<a name="method_getBic" class="anchor"></a>
#### public getBic() : string|null

```php
public getBic() : string|null
```

**Summary**

Возвращает bic.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md)

**Returns:** string|null - 


<a name="method_getCorrespondentAccount" class="anchor"></a>
#### public getCorrespondentAccount() : string|null

```php
public getCorrespondentAccount() : string|null
```

**Summary**

Возвращает correspondent_account.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md)

**Returns:** string|null - 


<a name="method_getName" class="anchor"></a>
#### public getName() : string|null

```php
public getName() : string|null
```

**Summary**

Возвращает name.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md)

**Returns:** string|null - 


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


<a name="method_setAccount" class="anchor"></a>
#### public setAccount() : self

```php
public setAccount(string|null $account = null) : self
```

**Summary**

Устанавливает account.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | account  | Счет получателя в банке. |

**Returns:** self - 


<a name="method_setBic" class="anchor"></a>
#### public setBic() : self

```php
public setBic(string|null $bic = null) : self
```

**Summary**

Устанавливает bic.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | bic  | БИК банка получателя. |

**Returns:** self - 


<a name="method_setCorrespondentAccount" class="anchor"></a>
#### public setCorrespondentAccount() : self

```php
public setCorrespondentAccount(string|null $correspondent_account = null) : self
```

**Summary**

Устанавливает correspondent_account.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | correspondent_account  | Корреспондентский счет банка получателя. |

**Returns:** self - 


<a name="method_setName" class="anchor"></a>
#### public setName() : self

```php
public setName(string|null $name = null) : self
```

**Summary**

Устанавливает name.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | name  | Название банка получателя. |

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
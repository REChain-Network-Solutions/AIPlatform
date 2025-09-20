# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\Airline
### Namespace: [\YooKassa\Request\Payments](../namespaces/yookassa-request-payments.md)
---
**Summary:**

Класс, представляющий модель Airline.

**Description:**

Объект с данными для [продажи авиабилетов](/developers/payment-acceptance/scenario-extensions/airline-tickets).
Используется только для платежей банковской картой.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$booking_reference](../classes/YooKassa-Request-Payments-Airline.md#property_booking_reference) |  | Номер бронирования. Обязателен на этапе создания платежа |
| public | [$bookingReference](../classes/YooKassa-Request-Payments-Airline.md#property_bookingReference) |  | Номер бронирования. Обязателен на этапе создания платежа |
| public | [$legs](../classes/YooKassa-Request-Payments-Airline.md#property_legs) |  | Список маршрутов |
| public | [$passengers](../classes/YooKassa-Request-Payments-Airline.md#property_passengers) |  | Список пассажиров |
| public | [$ticket_number](../classes/YooKassa-Request-Payments-Airline.md#property_ticket_number) |  | Уникальный номер билета. Обязателен на этапе подтверждения платежа |
| public | [$ticketNumber](../classes/YooKassa-Request-Payments-Airline.md#property_ticketNumber) |  | Уникальный номер билета. Обязателен на этапе подтверждения платежа |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [addLeg()](../classes/YooKassa-Request-Payments-Airline.md#method_addLeg) |  | Добавляет объект-контейнер с данными о маршруте. |
| public | [addPassenger()](../classes/YooKassa-Request-Payments-Airline.md#method_addPassenger) |  | Добавляет пассажира. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getBookingReference()](../classes/YooKassa-Request-Payments-Airline.md#method_getBookingReference) |  | Возвращает booking_reference. |
| public | [getLegs()](../classes/YooKassa-Request-Payments-Airline.md#method_getLegs) |  | Возвращает legs. |
| public | [getPassengers()](../classes/YooKassa-Request-Payments-Airline.md#method_getPassengers) |  | Возвращает список пассажиров. |
| public | [getTicketNumber()](../classes/YooKassa-Request-Payments-Airline.md#method_getTicketNumber) |  | Возвращает ticket_number. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [notEmpty()](../classes/YooKassa-Request-Payments-Airline.md#method_notEmpty) |  | Проверка на наличие данных. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setBookingReference()](../classes/YooKassa-Request-Payments-Airline.md#method_setBookingReference) |  | Устанавливает booking_reference. |
| public | [setLegs()](../classes/YooKassa-Request-Payments-Airline.md#method_setLegs) |  | Устанавливает legs. |
| public | [setPassengers()](../classes/YooKassa-Request-Payments-Airline.md#method_setPassengers) |  | Устанавливает список пассажиров. |
| public | [setTicketNumber()](../classes/YooKassa-Request-Payments-Airline.md#method_setTicketNumber) |  | Устанавливает ticket_number. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Payments/Airline.php](../../lib/Request/Payments/Airline.php)
* Package: YooKassa\Request
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Request\Payments\Airline
* Implements:
  * [\YooKassa\Request\Payments\AirlineInterface](../classes/YooKassa-Request-Payments-AirlineInterface.md)

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
<a name="property_booking_reference"></a>
#### public $booking_reference : string
---
***Description***

Номер бронирования. Обязателен на этапе создания платежа

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_bookingReference"></a>
#### public $bookingReference : string
---
***Description***

Номер бронирования. Обязателен на этапе создания платежа

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_legs"></a>
#### public $legs : \YooKassa\Common\ListObjectInterface|\YooKassa\Request\Payments\LegInterface[]
---
***Description***

Список маршрутов

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Request\Payments\LegInterface[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Request\Payments\LegInterface[]">LegInterface[]</abbr></a>

**Details:**


<a name="property_passengers"></a>
#### public $passengers : \YooKassa\Common\ListObjectInterface|\YooKassa\Request\Payments\PassengerInterface[]
---
***Description***

Список пассажиров

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Request\Payments\PassengerInterface[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Request\Payments\PassengerInterface[]">PassengerInterface[]</abbr></a>

**Details:**


<a name="property_ticket_number"></a>
#### public $ticket_number : string
---
***Description***

Уникальный номер билета. Обязателен на этапе подтверждения платежа

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_ticketNumber"></a>
#### public $ticketNumber : string
---
***Description***

Уникальный номер билета. Обязателен на этапе подтверждения платежа

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


<a name="method_addLeg" class="anchor"></a>
#### public addLeg() : self

```php
public addLeg(array|\YooKassa\Request\Payments\LegInterface $value) : self
```

**Summary**

Добавляет объект-контейнер с данными о маршруте.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Airline](../classes/YooKassa-Request-Payments-Airline.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\Payments\LegInterface</code> | value  | Объект-контейнер с данными о маршруте |

**Returns:** self - 


<a name="method_addPassenger" class="anchor"></a>
#### public addPassenger() : self

```php
public addPassenger(array|\YooKassa\Request\Payments\PassengerInterface $value) : self
```

**Summary**

Добавляет пассажира.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Airline](../classes/YooKassa-Request-Payments-Airline.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\Payments\PassengerInterface</code> | value  | Пассажир |

**Returns:** self - 


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


<a name="method_getBookingReference" class="anchor"></a>
#### public getBookingReference() : string|null

```php
public getBookingReference() : string|null
```

**Summary**

Возвращает booking_reference.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Airline](../classes/YooKassa-Request-Payments-Airline.md)

**Returns:** string|null - 


<a name="method_getLegs" class="anchor"></a>
#### public getLegs() : \YooKassa\Request\Payments\LegInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getLegs() : \YooKassa\Request\Payments\LegInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает legs.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Airline](../classes/YooKassa-Request-Payments-Airline.md)

**Returns:** \YooKassa\Request\Payments\LegInterface[]|\YooKassa\Common\ListObjectInterface - 


<a name="method_getPassengers" class="anchor"></a>
#### public getPassengers() : \YooKassa\Request\Payments\PassengerInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getPassengers() : \YooKassa\Request\Payments\PassengerInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает список пассажиров.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Airline](../classes/YooKassa-Request-Payments-Airline.md)

**Returns:** \YooKassa\Request\Payments\PassengerInterface[]|\YooKassa\Common\ListObjectInterface - Список пассажиров.


<a name="method_getTicketNumber" class="anchor"></a>
#### public getTicketNumber() : string|null

```php
public getTicketNumber() : string|null
```

**Summary**

Возвращает ticket_number.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Airline](../classes/YooKassa-Request-Payments-Airline.md)

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


<a name="method_notEmpty" class="anchor"></a>
#### public notEmpty() : bool

```php
public notEmpty() : bool
```

**Summary**

Проверка на наличие данных.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Airline](../classes/YooKassa-Request-Payments-Airline.md)

**Returns:** bool - 


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


<a name="method_setBookingReference" class="anchor"></a>
#### public setBookingReference() : self

```php
public setBookingReference(string|null $booking_reference = null) : self
```

**Summary**

Устанавливает booking_reference.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Airline](../classes/YooKassa-Request-Payments-Airline.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | booking_reference  | Номер бронирования. Обязателен, если не передан `ticket_number`. |

**Returns:** self - 


<a name="method_setLegs" class="anchor"></a>
#### public setLegs() : self

```php
public setLegs(\YooKassa\Common\ListObjectInterface|array|null $legs = null) : self
```

**Summary**

Устанавливает legs.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Airline](../classes/YooKassa-Request-Payments-Airline.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Common\ListObjectInterface OR array OR null</code> | legs  | Список перелетов. |

**Returns:** self - 


<a name="method_setPassengers" class="anchor"></a>
#### public setPassengers() : self

```php
public setPassengers(\YooKassa\Common\ListObjectInterface|array|null $passengers = null) : self
```

**Summary**

Устанавливает список пассажиров.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Airline](../classes/YooKassa-Request-Payments-Airline.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Common\ListObjectInterface OR array OR null</code> | passengers  | Список пассажиров. |

**Returns:** self - 


<a name="method_setTicketNumber" class="anchor"></a>
#### public setTicketNumber() : self

```php
public setTicketNumber(string|null $ticket_number = null) : self
```

**Summary**

Устанавливает ticket_number.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Airline](../classes/YooKassa-Request-Payments-Airline.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | ticket_number  | Уникальный номер билета. Если при создании платежа вы уже знаете номер билета, тогда `ticket_number` — обязательный параметр. Если не знаете, тогда вместо `ticket_number` необходимо передать `booking_reference` с номером бронирования. |

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
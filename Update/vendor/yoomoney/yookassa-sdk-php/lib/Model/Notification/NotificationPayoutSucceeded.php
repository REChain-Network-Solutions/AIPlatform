<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Model\Notification;

use Exception;
use YooKassa\Common\Exceptions\EmptyPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueException;
use YooKassa\Model\Payout\Payout;
use YooKassa\Model\Payout\PayoutInterface;
use YooKassa\Request\Payouts\PayoutResponse;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс объекта, присылаемого API при изменении статуса выплаты на "succeeded".
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @example 03-notification.php 3 Пример скрипта обработки уведомления
 *
 * @property PayoutInterface $object Объект с информацией о выплате
 */
class NotificationPayoutSucceeded extends AbstractNotification
{
    /**
     * Объект выплаты, для которого пришла нотификация. Так как нотификация может быть сгенерирована и поставлена в
     * очередь на отправку гораздо раньше, чем она будет получена на сайте, то опираться на статус пришедшей
     * выплаты не стоит, лучше запросить текущую информацию о выплате у API.
     *
     * @var PayoutInterface Объект платежа
     */
    #[Assert\NotBlank]
    #[Assert\Type(PayoutResponse::class)]
    #[Assert\Valid]
    private PayoutInterface $_object;

    /**
     * Конструктор объекта нотификации.
     *
     * Инициализирует текущий объект из ассоциативного массива, который просто путём JSON десериализации получен из
     * тела пришедшего запроса. При конструировании проверяется валидность типа передаваемого уведомления, если
     * передать уведомление не того типа, будет сгенерировано исключение типа {@link InvalidPropertyValueException}
     *
     * @param array $sourceArray Ассоциативный массив с информацией об уведомлении
     *
     * @throws Exception|InvalidPropertyValueException Генерируется если значение типа нотификации или события не равны
     *                                                 "notification" и "payout.succeeded" соответственно, что может говорить о том, что переданные в
     *                                                 конструктор данные не являются уведомлением нужного типа.
     */
    public function fromArray(iterable $sourceArray): void
    {
        $this->setType(NotificationType::NOTIFICATION);
        $this->setEvent(NotificationEventType::PAYOUT_SUCCEEDED);
        if (!empty($sourceArray['type'])) {
            if ($this->getType() !== $sourceArray['type']) {
                throw new InvalidPropertyValueException(
                    'Invalid value for "type" parameter in Notification',
                    0,
                    'notification.type',
                    $sourceArray['type']
                );
            }
        }
        if (!empty($sourceArray['event'])) {
            if ($this->getEvent() !== $sourceArray['event']) {
                throw new InvalidPropertyValueException(
                    'Invalid value for "event" parameter in Notification',
                    0,
                    'notification.event',
                    $sourceArray['event']
                );
            }
        }
        if (empty($sourceArray['object'])) {
            throw new EmptyPropertyValueException('Parameter object in NotificationPayoutSucceeded is empty');
        }
        $this->setObject($sourceArray['object']);
    }

    /**
     * Возвращает объект с информацией о выплате, уведомление о которой хранится в текущем объекте.
     *
     * Так как нотификация может быть сгенерирована и поставлена в очередь на отправку гораздо раньше, чем она будет
     * получена на сайте, то опираться на статус пришедшей выплаты не стоит, лучше запросить текущую информацию о
     * выплате у API.
     *
     * @return PayoutInterface Объект с информацией о выплате
     */
    public function getObject(): PayoutInterface
    {
        return $this->_object;
    }

    /**
     * Устанавливает объект с информацией о выплате, уведомление о которой хранится в текущем объекте.
     *
     * @param PayoutInterface|array $object
     *
     * @return self
     */
    public function setObject(mixed $object): self
    {
        $this->_object = $this->validatePropertyValue('_object', $object);
        return $this;
    }
}

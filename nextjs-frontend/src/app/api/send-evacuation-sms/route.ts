import { NextResponse } from 'next/server';

export async function POST() {
  const twilioAccountSid = process.env.TWILIO_ACCOUNT_SID || '';
  const twilioAuthToken = process.env.TWILIO_AUTH_TOKEN || '';
  const twilioUrl = `https://api.twilio.com/2010-04-01/Accounts/${twilioAccountSid}/Messages.json`;

  try {
    const smsBody = new URLSearchParams();
    smsBody.append('To', '+918828642788');
    smsBody.append('MessagingServiceSid', process.env.TWILIO_MESSAGING_SERVICE_SID || '');
    smsBody.append('Body', 'METHANE GAS DETECTED! SAFTEY EVACUATION NEEDED');

    const response = await fetch(twilioUrl, {
      method: 'POST',
      headers: {
        'Authorization': 'Basic ' + Buffer.from(`${twilioAccountSid}:${twilioAuthToken}`).toString('base64'),
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: smsBody.toString(),
    });

    const data = await response.json();

    if (!response.ok) {
      console.error('Twilio API error:', data);
      return NextResponse.json({ success: false, error: data }, { status: response.status });
    }

    console.log('SMS sent successfully:', data.sid);
    return NextResponse.json({ success: true, messageSid: data.sid });
  } catch (error) {
    console.error('Failed to send SMS:', error);
    return NextResponse.json({ success: false, error: 'Failed to send SMS' }, { status: 500 });
  }
}
